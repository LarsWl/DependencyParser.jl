export zero_cost_transitions, transition_costs, optimal_transition_index, gold_scores
using Flux

function zero_cost_transitions(state::GoldState)
  filter(trans -> is_valid(state.config, trans.move) && cost(state, trans.move, trans.label) <= 0, state.system.transitions)
end

function gold_scores(costs::Vector{Int64})
  map(costs) do cost 
    if cost <= 0
      1
    elseif cost == FORBIDDEN_COST
      -1
    else
      0
    end
  end |> vec -> reshape(vec, 1, :)
end

function optimal_transition_index(costs::Vector{Int64}, system::ArcEagerSystem)
  min_index = 1
  
  foreach(enumerate(costs)) do (index, cost)
    move_code_mame = system.transitions[index].move.code_name
    if cost < costs[min_index] || cost == costs[min_index] && (move_code_mame == Moves.LEFT || move_code_mame == Moves.RIGHT)
      min_index = index
    end
  end

  min_index
end

function transition_costs(state::GoldState)
  map(state.system.transitions) do trans
    is_valid(state.config, trans.move) ? cost(state, trans.move, trans.label) : FORBIDDEN_COST
  end
end
