export Configuration;
export stack_depth, buffer_length, config_push!, config_pop!, unshift!, add_arc, is_unshiftable, set_reshiftable;
export has_head, del_arc;

export NONEXIST_TOKEN, UNKNOWN_TOKEN, NULL_TOKEN, ROOT_TOKEN

using DataStructures

const NONEXIST_TOKEN = -1
const UNKNOWN_TOKEN = "-UNKNOWN-"
const NULL_TOKEN = "-NULL-"
const LIMIT_TRANSITIONS_NUMBER = 500

mutable struct Configuration
  buffer::Vector{Integer}
  stack::Stack{Integer}
  sentence::Sentence
  unshiftable::Vector{Bool}
  tree::DependencyTree


  Configuration(
    buffer::Vector{Integer},
    stack::Stack{Integer},
    sentence::Sentence,
    unshiftable::Vector{Bool},
    tree::DependencyTree
  ) = new(buffer, stack, sentence, unshiftable, tree)
  
  function Configuration(sentence::Sentence)
    buffer = collect(1:sentence.length)
    unshiftable = zeros(Bool, sentence.length)
    stack = Stack{Integer}()
    push!(stack, 0)
    tree = DependencyTree(sentence)

    new(buffer, stack, sentence, unshiftable, tree)
  end
end

stack_depth(config::Configuration) = length(config.stack)

buffer_length(config::Configuration) = length(config.buffer)

function config_push!(config::Configuration)
  buffer_length(config) == 0 && return

  buffer_element = popfirst!(config.buffer)
  push!(config.stack, buffer_element)
end

function config_pop!(config::Configuration)
  stack_depth(config) == 0 && return

  pop!(config.stack)
end

function unshift!(config::Configuration)
  word_id = config_pop!(config)
  if word_id === nothing
    return
  end

  insert!(config.buffer, 1, word_id)
  config.unshiftable[word_id] = true
end

function add_arc(config::Configuration, head_id::Integer, word_id::Integer, label::String)
  set_arc(config.tree, word_id, head_id, label)
end

function is_unshiftable(config::Configuration, word_id::Integer)
  1 <= word_id <= length(config.unshiftable) ? config.unshiftable[word_id] : true
end


function set_reshiftable(config::Configuration, word_id::Integer)
  if 1 <= word_id <= length(config.unshiftable)
    config.unshiftable[word_id] = false 
  end
end

function get_token(config::Configuration, index::Integer)
  index == 0 && return ROOT_TOKEN

  1 <= index <= config.sentence.length && return config.sentence.tokens[index].name

  NULL_TOKEN
end

function get_tag(config::Configuration, index::Integer)
  index == 0 && return ROOT_TOKEN
  
  1 <= index <= config.sentence.length && return config.sentence.pos_tags[index].name

  NULL_TOKEN
end

function get_label(config::Configuration, index::Integer)
  1 <= index <= config.sentence.length && return config.tree.nodes[index].label

  NULL_TOKEN
end

function get_buffer_element(config::Configuration, index::Integer)
  1 <= index <= buffer_length(config) && return config.buffer[index]

  NONEXIST_TOKEN
end

function get_stack_element(config::Configuration, index::Integer)
  stack_vector = collect(config.stack)
  1 <= index <= stack_depth(config) && return stack_vector[index]

  NONEXIST_TOKEN
end

function is_terminal(config::Configuration)
  stack_depth(config) == 1 && buffer_length(config) == 0
end