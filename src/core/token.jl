export Token

struct Token
  id::Integer
  name::String
end

function is_sent_start(token::Token)
  token.id == 1
end