export zero_cost

function zero_cost(state::GoldState)
  valid_moves(state.config) |> filter(move -> cost(state, move) == 0, moves)
end