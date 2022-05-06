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
    if cost == FORBIDDEN_COST
      -1
    elseif cost <= 0
      1
    else
      0
    end
  end
end