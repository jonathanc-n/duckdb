#include "core_functions/aggregate/distributive_functions.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/null_value.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"

namespace duckdb {

namespace {

// generic state that works with any type
struct FirstChangeState {
	bool is_initialized;
	bool change_detected;
	Value first_value;
	Value changed_value;
	Value change_time;
};

struct FirstChangeOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.is_initialized = false;
		state.change_detected = false;
		state.first_value = Value();
		state.changed_value = Value();
		state.change_time = Value();
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &aggr_input_data) {
		state.first_value = Value();
		state.changed_value = Value();
		state.change_time = Value();
	}

	template <class STATE>
	static void Update(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
	                   idx_t count) {
		D_ASSERT(input_count == 2);
		auto &value_vec = inputs[0];
		auto &time_vec = inputs[1];

		UnifiedVectorFormat value_format, time_format, state_format;
		value_vec.ToUnifiedFormat(count, value_format);
		time_vec.ToUnifiedFormat(count, time_format);
		state_vector.ToUnifiedFormat(count, state_format);

		auto states = UnifiedVectorFormat::GetData<STATE *>(state_format);

		for (idx_t i = 0; i < count; i++) {
			const auto state_idx = state_format.sel->get_index(i);
			auto &state = *states[state_idx];

			const auto value_idx = value_format.sel->get_index(i);
			const auto time_idx = time_format.sel->get_index(i);

			const bool value_null = !value_format.validity.RowIsValid(value_idx);
			const bool time_null = !time_format.validity.RowIsValid(time_idx);

			if (!state.is_initialized) {
				// first value we see, so just initialize the state
				if (!value_null) {
					state.first_value = value_vec.GetValue(value_idx);
				} else {
					state.first_value = Value();
				}
				state.is_initialized = true;
				continue;
			}

			if (state.change_detected) {
				// we already found a change, so we can skip the rest
				continue;
			}

			// check if the value actually changed
			bool value_changed = false;
			if (value_null != state.first_value.IsNull()) {
				// null status changed, so that counts as a change
				value_changed = true;
			} else if (!value_null && !state.first_value.IsNull()) {
				// both are non-null, so compare the actual values
				Value current_value = value_vec.GetValue(value_idx);
				if (current_value != state.first_value) {
					value_changed = true;
				}
			}

			if (value_changed) {
				state.change_detected = true;
				if (!value_null) {
					state.changed_value = value_vec.GetValue(value_idx);
				} else {
					state.changed_value = Value();
				}
				if (!time_null) {
					state.change_time = time_vec.GetValue(time_idx);
				} else {
					state.change_time = Value();
				}
			}
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &aggr_input_data) {
		if (!source.is_initialized) {
			return;
		}

		if (!target.is_initialized) {
			// target isn't initialized yet, so just copy everything from source
			target = source;
			return;
		}

		// if target already found a change, keep it since it's earlier in the sorted order
		if (target.change_detected) {
			return;
		}

		// if source found a change, use that one
		if (source.change_detected) {
			target.change_detected = true;
			target.changed_value = source.changed_value;
			target.change_time = source.change_time;
			return;
		}

		// neither found a change yet, so check if there's a change at the boundary between them
		bool value_changed = false;
		if (target.first_value.IsNull() != source.first_value.IsNull()) {
			value_changed = true;
		} else if (!target.first_value.IsNull() && !source.first_value.IsNull()) {
			if (target.first_value != source.first_value) {
				value_changed = true;
			}
		}

		if (value_changed) {
			// the change happens at the boundary, so use source's first value as the changed value
			target.change_detected = true;
			target.changed_value = source.first_value;
			// we don't have a timestamp for the boundary, so just use null
			target.change_time = Value();
		}
	}

	template <class STATE>
	static void Finalize(STATE &state, AggregateFinalizeData &finalize_data) {
		if (!state.is_initialized || !state.change_detected) {
			finalize_data.ReturnNull();
			return;
		}

		// create a struct with the timestamp and value
		child_list_t<Value> struct_values;
		struct_values.push_back(make_pair("timestamp", state.change_time));
		struct_values.push_back(make_pair("value", state.changed_value));

		Value struct_value = Value::STRUCT(struct_values);
		finalize_data.result.SetValue(finalize_data.result_idx, struct_value);
	}

	static bool IgnoreNull() {
		return false;
	}
};

// generic bind function that works with any types
unique_ptr<FunctionData> FirstChangeBind(ClientContext &context, AggregateFunction &function,
                                         vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() < 1) {
		throw BinderException("first_change takes at least one argument");
	}

	auto value_type = arguments[0]->return_type;
	
	// the ORDER BY column gets added as a second argument by the sorted aggregate wrapper
	// when we first bind, we only have the value argument
	// after the sorted aggregate wrapper does its thing, we'll have value + ORDER BY column
	LogicalType time_type = LogicalType::TIMESTAMP; // default placeholder
	
	if (arguments.size() >= 2) {
		time_type = arguments[1]->return_type;
		// update the function signature to include both arguments (value + time)
		function.arguments = {value_type, time_type};
	}
	// note: when we only have 1 argument, don't mess with function.arguments
	// the sorted aggregate wrapper will add the ORDER BY column and handle everything
	
	// make sure both types are valid - use placeholders if we need to
	// the sorted aggregate wrapper needs a valid return type, so we gotta set one
	if (time_type.id() == LogicalTypeId::ANY || time_type.id() == LogicalTypeId::INVALID) {
		time_type = LogicalType::TIMESTAMP; // use a valid placeholder
	}
	// use a placeholder for value_type if it's still ANY (will get updated when called with concrete types)
	LogicalType struct_value_type = value_type;
	if (value_type.id() == LogicalTypeId::ANY || value_type.id() == LogicalTypeId::INVALID) {
		struct_value_type = LogicalType::INTEGER; // use a valid placeholder
	}
	
	// create the struct return type - both types need to be valid (no ANY or INVALID)
	// the sorted aggregate wrapper will use this return type, so it better be valid
	function.SetReturnType(LogicalType::STRUCT({make_pair("timestamp", time_type), 
	                                             make_pair("value", struct_value_type)}));

	// use the generic implementation
	using STATE = FirstChangeState;
	using OP = FirstChangeOperation;

	function.SetStateSizeCallback(AggregateFunction::StateSize<STATE>);
	function.SetStateInitCallback(AggregateFunction::StateInitialize<STATE, OP, AggregateDestructorType::LEGACY>);
	function.SetStateUpdateCallback(OP::template Update<STATE>);
	function.SetStateCombineCallback(AggregateFunction::StateCombine<STATE, OP>);
	function.SetStateFinalizeCallback(AggregateFunction::StateVoidFinalize<STATE, OP>);
	function.SetStateDestructorCallback(AggregateFunction::StateDestroy<STATE, OP>);

	return nullptr;
}

} // namespace

AggregateFunctionSet FirstChangeFun::GetFunctions() {
	AggregateFunctionSet fun;

	// register with ANY return type - the bind function will set the proper struct return type
	// this matches the pattern used by first/last functions
	AggregateFunction generic_function({LogicalType::ANY}, LogicalType::ANY, nullptr, nullptr, nullptr, nullptr, nullptr,
	                                   nullptr, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, FirstChangeBind);
	generic_function.SetOrderDependent(AggregateOrderDependent::ORDER_DEPENDENT);
	fun.AddFunction(generic_function);

	return fun;
}

} // namespace duckdb

