export zero_cost_transitions, transition_costs, gold_scores
using Flux

function zero_cost_transitions(state::GoldState)
  filter(trans -> is_valid(state.config, trans.move) && cost(state, trans.move, trans.label) <= 0, state.system.transitions)
end


# I inverse cost& because I need to zero cost transitions have best value after softmax
function transition_costs(state::GoldState)
  map(state.system.transitions) do trans
    is_valid(state.config, trans.move) ? cost(state, trans.move, trans.label) : FORBIDDEN_COST
  end
end

function gold_scores(costs::Vector{Int64})
  map(costs) do cost 
    if cost <= 0
      [1, 0]
    else
      [0, 1]
    end
  end |> scores -> hcat(scores...) # .* weight_mask(costs)
end

function weight_mask(costs)
  shift_weights = [1, 1, 2]
  other_weights = [[2, 1, 2] for i in 1:(length(costs) - 1)]

  hcat(shift_weights, other_weights...)
end