export zero_cost_transitions, transition_costs

function zero_cost_transitions(state::GoldState)
  filter(trans -> is_valid(state.config, trans.move) && cost(state, trans.move, trans.label) <= 0, state.system.transitions)
end


# I inverse cost& because I need to zero cost transitions have best value after softmax
function transition_costs(state::GoldState)
  base_costs = map(trans -> is_valid(state.config, trans.move) ? cost(state, trans.move, trans.label) : FORBIDDEN_COST, state.system.transitions)

  map(cost -> Float32(cost * -1), base_costs)
end