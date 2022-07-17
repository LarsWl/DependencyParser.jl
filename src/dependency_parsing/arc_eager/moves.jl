export Shift, Reduce, RightArc, LeftArc, Break
export cost, is_valid, transition, valid_moves

using ..DependencyParsing

module Moves
  const SHIFT = "S"
  const REDUCE = "D"
  const RIGHT = "R"
  const LEFT = "L"
end

struct Shift <: Move
  code_name::String
end
struct Reduce <: Move
  code_name::String
end
struct RightArc <: Move
  code_name::String
end
struct LeftArc <: Move
  code_name::String
end

#=
  Cost: push_cost 
=#
function cost(state::GoldState, move::Shift, label::String)
  cost = 0
  if head_state(state, state.config.buffer[begin]) == HEAD_IN_STACK &&
    state.gold_tree.nodes[state.config.buffer[begin]].head_id != first(state.config.stack)
    cost += 1
  end

  cost += dependents_in_stack_count(state, state.config.buffer[begin])

  cost
end

#=
  Validity:
  * If stack is empty
  * At least two words in sentence
  * Word has not been shifted before
=#
function is_valid(config::Configuration, move::Shift)
  if stack_depth(config) == 0
    return true
  elseif buffer_length(config) < 2
    return false
  elseif is_unshiftable(config, config.buffer[begin])
    return false
  else
    return true
  end
end

# Move the first word of the buffer onto the stack and mark it as "shifted"
function transition(config::Configuration, label::String, move::Shift)
  config_push!(config)
end

#=
Cost:
  * If B[0] is the start of a sentence, cost is 0
  * Arcs between stack and buffer
  * If arc has no head, we're saving arcs between S[0] and S[1:], so decrement
      cost by those arcs.
=#
function cost(state::GoldState, move::Reduce, label::String)
  cost = 0

  cost += dependents_in_buffer_count(state, first(state.config.stack))
  if head_state(state, first(state.config.stack)) == HEAD_IN_BUFFER
    cost += 1
  end

  if !has_head(state.config.tree, first(state.config.stack))
    if head_state(state, first(state.config.stack)) == HEAD_IN_STACK
      cost -= 1
    end

    cost -= dependents_in_stack_count(state, first(state.config.stack))
  end

  cost
end

#=
  Validity:
    * Stack not empty
=#
function is_valid(config::Configuration, move::Reduce)
  stack_depth(config) == 0 && return false
    
  true
end

# Pop from the stack. If it has no head and the stack isn't empty, place it back on the buffer.
function transition(config::Configuration, label::String, move::Reduce)
  if has_head(config.tree, first(config.stack)) || stack_depth(config) == 1
    config_pop!(config)
  else
    unshift!(config)
  end
end

# push_cost + (not shifted[b0] and Arc(B[1:], B[0])) - Arc(S[0], B[0], label): NOT VALID COMMENT
function cost(state::GoldState, move::RightArc, label::String)
  cost = 0
  stack = state.config.stack |> collect
  s0 = stack[begin]
  b0 = state.config.buffer[begin]
  b0_gold_head = state.gold_tree.nodes[b0].head_id

  cost += dependents_in_stack_count(state, b0)
  if head_state(state, b0) == HEAD_IN_STACK
    cost += 1
  end

  if b0_gold_head == s0
    cost -= 1
    if !label_correct(state, b0, label)
      cost += 1
    end
  elseif head_state(state, b0) == HEAD_IN_BUFFER && !state.config.unshiftable[b0]
    cost += 1
  end

  cost
end
#=
  Validity:
  * len(S) >= 1
  * len(B) >= 1
  * not is_sent_start(B[0]) ?
=#
function is_valid(config::Configuration, move::RightArc)
  if stack_depth(config) < 1
    return false
  elseif buffer_length(config) < 1
    return false
  else
    return true
  end
    
end

# Add an arc from S[0] to B[0]. Push B[0].
function transition(config::Configuration, label::String, move::RightArc)
  dependent = config.buffer[begin]
  head = first(config.stack)
  add_arc(config, head, dependent, label)
  config_push!(config)
end

# pop_cost - Arc(B[0], S[0], label) + (Arc(S[1], S[0]) if H(S[0]) else Arcs(S, S[0])): NOT VALID COMMENT
function cost(state::GoldState, move::LeftArc, label::String)
  cost = 0
  stack = state.config.stack |> collect
  s0 = stack[begin]
  b0 = state.config.buffer[begin]
  s0_gold_head = s0 == 0 ? EMPTY_NODE : state.gold_tree.nodes[s0].head_id

  cost += dependents_in_buffer_count(state, s0)
  if head_state(state, s0) == HEAD_IN_BUFFER
    cost += 1
  end

  if has_head(state.config.tree, s0)
    if stack_depth(state.config) >= 2 && s0_gold_head == stack[begin + 1]
      cost += 1
    end
  else
    if head_state(state, s0) == HEAD_IN_STACK
      cost += 1
    end
    cost += dependents_in_stack_count(state, s0)
  end
  
  if s0_gold_head == b0
    cost -= 1
    if !label_correct(state, s0, label)
      cost += 1
    end
  end

  cost
end

#=
Validity:
  * len(S) >= 1
  * len(B) >= 1
  * not is_sent_start(B[0]) ?
=#
function is_valid(config::Configuration, move::LeftArc)
  if stack_depth(config) < 1
    return false
  elseif buffer_length(config) < 1
    return false
  else
    return true
  end
end

# Add an arc between B[0] and S[0], replacing the previous head of S[0] ifone is set. Pop S[0] from the stack.
function transition(config::Configuration, label::String, move::LeftArc)
  head = config.buffer[begin]
  dependent = config_pop!(config)
  has_head(config.tree, dependent) && del_arc(config.tree, dependent)
  add_arc(config, head, dependent, label)
  set_reshiftable(config, dependent)
end
