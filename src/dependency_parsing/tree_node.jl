export TreeNode;
export EMPTY_LABEL, EMPTY_NODE, ROOT_TOKEN, NO_HEAD

const ROOT_TOKEN = "-ROOT-"
const NO_HEAD = "-NO_HEAD-"
const EMPTY_LABEL = "EMPTY_LABEL"
const EMPTY_NODE = -1 

mutable struct TreeNode
  word_id::Integer
  label::String
  head_id::Integer
  token::String

  TreeNode(word_id::Integer) = new(word_id, EMPTY_LABEL, EMPTY_NODE, ROOT_TOKEN)
  TreeNode(word_id::Integer, token::String) = new(word_id, EMPTY_LABEL, EMPTY_NODE, token)
  TreeNode(word_id::Integer, label::String, head_id::Integer) = new(word_id, label, head_id)
end