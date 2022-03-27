export GoldState;
export push_cost, pop_cost, update_state;
export HEAD_ASSIGNED, HEAD_IN_BUFFER, HEAD_IN_STACK, HEAD_UNKNOWN;

const FORBIDDEN_COST = 9000

const HEAD_UNKNOWN = 0
const HEAD_ASSIGNED = 1
const HEAD_IN_STACK = 2
const HEAD_IN_BUFFER = 3

mutable struct GoldState
  gold_tree::DependencyTree
  root_dependents_in_stack_count::Integer
  root_dependents_in_buffer_count::Integer
  dependents_in_stack_count::Vector{Integer}
  dependents_in_buffer_count::Vector{Integer}
  heads_states::Vector{Integer}
  config::Configuration

  function GoldState(gold_tree::DependencyTree, config::Configuration) 
    state = new(gold_tree)
    update_state(state, config)

    state
  end
end

function update_state(state::GoldState, config::Configuration)
  state.config = config
  state.dependents_in_stack_count = zeros(state.gold_tree.length)
  state.dependents_in_buffer_count = zeros(state.gold_tree.length)
  state.root_dependents_in_buffer_count = 0
  state.root_dependents_in_stack_count = 0
  state.heads_states = zeros(state.gold_tree.length)

  # calculate dependents for root separatly
  foreach(config.buffer) do dependent
    if arc_present(state.gold_tree, 0, dependent)
      state.root_dependents_in_buffer_count += 1
    end
  end

  # calculate dependents for root separatly
  foreach(config.stack) do dependent
    if arc_present(state.gold_tree, 0, dependent)
      state.root_dependents_in_stack_count += 1
    end
  end

  # calculate dependents for each node
  for head = 1:state.gold_tree.length
    foreach(config.buffer) do dependent
      if arc_present(state.gold_tree, head, dependent)
        state.dependents_in_buffer_count[head] += 1
      end
    end

    foreach(config.stack) do dependent
      if arc_present(state.gold_tree, head, dependent)
        state.dependents_in_stack_count[head] += 1
      end
    end

    state.heads_states[head] = if head_in_stack(state, config, head)
      HEAD_IN_STACK
    elseif head_in_buffer(state, config, head)
      HEAD_IN_BUFFER
    elseif has_head(state.gold_tree, head)
      HEAD_ASSIGNED
    else
      HEAD_UNKNOWN
    end
  end
end

function push_cost(config::Configuration, gold_state::GoldState)
  cost = 0
  b0 = config.buffer[begin]
  if b0 < 1
    return FORBIDDEN_COST
  else
    foreach(config.stack) do word
      if is_arc_in_gold(gold_state.gold_tree, b0, word) || is_arc_in_gold(gold_state.gold_tree, word, b0)
        cost += 1
      end
    end
  end
end

function pop_cost(config::Configuration, gold_state::GoldState)
  cost = 0
  s0 = first(config.stack)

  if s0 < 1
    return FORBIDDEN_COST
  else
    foreach(config.buffer) do word
      if is_arc_in_gold(gold_state.gold_tree, s0, word) || is_arc_in_gold(gold_state.gold_tree, word, s0)
        cost += 1
      end
    end
  end
end

head_in_stack(state::GoldState, config::Configuration, word::Integer) = state.gold_tree.nodes[word].head_id in config.stack
head_in_buffer(state::GoldState, config::Configuration, word::Integer) = state.gold_tree.nodes[word].head_id in config.buffer