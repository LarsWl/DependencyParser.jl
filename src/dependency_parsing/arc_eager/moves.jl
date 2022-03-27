export Shift, Reduce, RightArc, LeftArc, Break
export cost, is_valid, transition

using ..DependencyParsing

module Moves
  const SHIFT = 'S'
  const REDUCE = 'D'
  const RIGHT = 'R'
  const LEFT = 'L'
  const BREAK = 'B'
end

struct Shift <: Move end;
struct Reduce <: Move end;
struct RightArc <: Move end;
struct LeftArc <: Move end;
struct Break <: Move end;

#=
  Cost: push_cost 
=#
function cost(state::GoldState, move::Shift, system::ArcEagerSystem)
  cost = 0
  if state.heads_states[state.config.buffer[begin]] == HEAD_IN_STACK &&
    state.gold_tree.nodes[state.config.buffer[begin]].head_id != first(state.config.stack)
    cost += 1
  end

  cost += state.dependents_in_stack_count[state.config.buffer[begin]]

  cost
end

#=
  Validity:
  * If stack is empty
  * At least two words in sentence
  * Word has not been shifted before
=#
function is_valid(config::Configuration, move::Shift, system::ArcEagerSystem)
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
function transition(config::Configuration, label::String, move::Shift, system::ArcEagerSystem)
  config_push!(config)
end

#=
Cost:
  * If B[0] is the start of a sentence, cost is 0
  * Arcs between stack and buffer
  * If arc has no head, we're saving arcs between S[0] and S[1:], so decrement
      cost by those arcs.
=#
function cost(state::GoldState, move::Reduce, system::ArcEagerSystem)
  cost = 0
  state.config.buffer[begin] == 0 && return cost

  cost += state.dependents_in_buffer_count[first(state.config.stack)]
  if state.heads_states[first(state.config.stack)] == HEAD_IN_BUFFER
    cost += 1
  end

  if !has_head(state.config.tree, first(state.config.stack))
    if state.heads_states[first(state.config.stack)] == HEAD_IN_STACK
      cost -= 1
    end

    cost -= state.dependents_in_stack_count[first(state.config.stack)]
  end

  cost
end

#=
  Validity:
    * Stack not empty
    * Buffer nt empty
    * Stack depth 1 and cannot senten start l_edge(st.B(0)) ?
=#
function is_valid(config::Configuration, move::Reduce, system::ArcEagerSystem)
  if stack_depth(config) == 0
    return false
  elseif buffer_length(config) == 0
    return false
  else
    return true
  end
end

# Pop from the stack. If it has no head and the stack isn't empty, place it back on the buffer.
function transition(config::Configuration, label::String, move::Reduce, system::ArcEagerSystem)
  if has_head(config.tree, first(config.stack)) || stack_depth(config) == 1
    config_pop!(config)
  else
    unshift!(config)
  end
end

# push_cost + (not shifted[b0] and Arc(B[1:], B[0])) - Arc(S[0], B[0], label): NOT VALID COMMENT
function cost(state::GoldState, move::RightArc, system::ArcEagerSystem)
  cost = 0
  b0 = state.config.buffer[begin]
  cost += state.dependents_in_stack_count[b0]
  if state.heads_states[b0] == HEAD_IN_STACK
    cost += 1
  elseif state.heads_states[b0] == HEAD_IN_BUFFER
    cost -= 1
  end

  cost
end

#=
  Validity:
  * len(S) >= 1
  * len(B) >= 1
  * not is_sent_start(B[0]) ?
=#
function is_valid(config::Configuration, move::RightArc, system::ArcEagerSystem)
  if stack_depth(config) < 1
    return false
  elseif buffer_length(config) < 1
    return false
  else
    return true
  end
    
end

# Add an arc from S[0] to B[0]. Push B[0].
function transition(config::Configuration, label::String, move::RightArc, system::ArcEagerSystem)
  dependent = config.buffer[begin]
  head = first(config.stack)
  add_arc(config, head, dependent, label)
  config_push!(config)
end

# pop_cost - Arc(B[0], S[0], label) + (Arc(S[1], S[0]) if H(S[0]) else Arcs(S, S[0])): NOT VALID COMMENT
function cost(state::GoldState, move::LeftArc, system::ArcEagerSystem)
  cost = 0
  s0 = first(state.config.stack)
  b0 = state.config.buffer[begin]
  cost += state.dependents_in_buffer_count[s0]
  if state.heads_states[s0] == HEAD_IN_BUFFER
    cost += 1
  end
  has_head(state.config.tree, s0) || return cost

  s0_head = state.config.tree.nodes[s0].head_id
  s0_gold_head = state.gold_tree.nodes[s0].head_id
  if s0_head == s0_gold_head
    cost += 1
  elseif (s0_gold_head in state.config.buffer) && s0_gold_head != b0
    cost += 1
  end

  return cost
end

#=
Validity:
  * len(S) >= 1
  * len(B) >= 1
  * not is_sent_start(B[0]) ?
=#
function is_valid(config::Configuration, move::LeftArc, system::ArcEagerSystem)
  if stack_depth(config) < 1
    return false
  elseif buffer_length(config) < 1
    return false
  else
    return true
  end
end

# Add an arc between B[0] and S[0], replacing the previous head of S[0] ifone is set. Pop S[0] from the stack.
function transition(config::Configuration, label::String, move::LeftArc, system::ArcEagerSystem)
  head = config.buffer[begin]
  dependent = config_pop!(config)
  add_arc(config, head, dependent, label)
  set_reshiftable(config, dependent)
end



# Not sure that i need this Move

function cost(config::Configuration, move::Break, system::ArcEagerSystem)
end

#=
Validity:
  * len(buffer) >= 2
  * B[1] == B[0] + 1
  * not is_sent_start(B[1])
  * not cannot_sent_start(B[1])
=#
function is_valid(config::Configuration, move::Break, system::ArcEagerSystem)
  if buffer_length(config) < 2
    return false
  elseif config.buffer[begin] != config.buffer[begin + 1]
    return false
  else
    return true
  end
end

#Mark the second word of the buffer as the start of a sentence. 
function transition(config::Configuration, label::String, move::Break, system::ArcEagerSystem)
end