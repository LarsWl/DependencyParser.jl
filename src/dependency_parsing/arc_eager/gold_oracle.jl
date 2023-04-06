export zero_cost_transitions, transition_costs, gold_scores
using Flux

function zero_cost_transitions(state::GoldState)
  filter(trans -> is_valid(state.config, trans.move) && cost(state, trans.move, trans.label) <= 0, state.system.transitions)
end

function gold_scores(costs::Vector{Int64})
  gold = zeros(Int8, length(costs))
  min_value = minimum(costs)
  gold[findlast(==(min_value), costs)] = 1

  Flux.onehot(1, gold)
end

function transition_costs(state::GoldState)
  map(state.system.transitions) do trans
    is_valid(state.config, trans.move) ? cost(state, trans.move, trans.label) : FORBIDDEN_COST
  end
end
