export DepParser
export execute_transition, form_batch, predict_transition

using ..DependencyParser.Units
import .ArcEager: GoldState
import .ArcEager: execute_transition, zero_cost

using TextAnalysis
using TextModels

struct DepParser <: AbstractDepParser
  settings::Settings
  parsing_system::ParsingSystem
  model::Model

  function DepParser(settings::Settings, model_file::String)
    model = Model(model_file)
    
    new(settings, ArcEager.ArcEagerSystem(), model)
  end

  DepParser(settings::Settings, model::Model, system::ParsingSystem) = new(settings, system, model)
end

function (parser::DepParser)(tokens_with_tags)
  println(tokens_with_tags)
end

function execute_transition(parser::DepParser, transition::Transition)
  execute_transition(parser.config, transition, parser.parsing_system)
end

function train!(system::ParsingSystem, train_file::String, embeddings_file::String)
  iterations = 10
  conllu_sentences = load_connlu_file(train_file)
  settings = Settings()

  model = Model(settings, system, embeddings_file, conllu_sentences)

  for i = 1:iterations
    for (string_doc, gold_tree) in corpus
      sentence = tokens(string_doc.text) |> pos |> Sentence

      config = Configuration(sentence)
      gold_state = GoldState(gold_tree, config)

      preticted_transition = predict(model, config)
      zero_cost_transitions = zero_cost(gold_state)

      if predicted_transition in zero_cost_transitions
        update_model(model)
      end

      transition = zero_cost_transitions[begin]

      execute_transition(config, transition, system)
    end
  end

  write!(model)
end


#=
while batch_size only is 48 there is structure of batch
1-18 - word_ids
19-36 - tag_ids
37-48 - label_ids
=#
const POS_OFFSET = 18
const LABEL_OFFSET = 36
const STACK_OFFSET = 6
const STACK_NUMBER = 6

function form_batch(parser::DepParser, config::Configuration)
  batch = zeros(Integer, parser.settings.batch_size)

  word_id_by_word_index(word_index::Integer) = get_token(config, word_index) |> token -> get_word_id(parser, token)
  tag_id_by_word_index(word_index::Integer) = get_tag(config, word_index) |> tag -> get_tag_id(parser, tag)
  label_id_by_word_index(word_index::Integer) = get_label(config, word_index) |> label -> get_label_id(parser, label)

  # add top three stack elements and top three buffers elems with their's tags
  for i = 1:3
    stack_word_index = get_stack_element(config, i)
    buffer_word_index = get_buffer_element(config, i)

    batch[i] = word_id_by_word_index(stack_word_index)
    batch[i + POS_OFFSET] = tag_id_by_word_index(stack_word_index)
    batch[i + 3] = word_id_by_word_index(buffer_word_index)
    batch[i + POS_OFFSET + 3] = tag_id_by_word_index(buffer_word_index)
  end

  #=
    Add: 
    The first and second leftmost / rightmost children of the top two words on the stack and
    The leftmost of leftmost / rightmost of rightmost children of the top two words on the stack
  =#
  for stack_id = 1:2
    stack_word_index = get_stack_element(config, stack_id)

    set_word_data_by_index_with_offset = function (word_index::Integer, additional_offset)
      batch[STACK_OFFSET + (stack_id - 1) * STACK_NUMBER + additional_offset] = word_id_by_word_index(word_index)
      batch[STACK_OFFSET + POS_OFFSET + (stack_id - 1) * STACK_NUMBER + additional_offset] = tag_id_by_word_index(word_index)
      batch[LABEL_OFFSET + (stack_id - 1) * STACK_NUMBER + additional_offset] = label_id_by_word_index(word_index)
    end

    get_left_child(config.tree, stack_word_index) |> word_index -> set_word_data_by_index_with_offset(word_index, 1)
    get_right_child(config.tree, stack_word_index) |> word_index -> set_word_data_by_index_with_offset(word_index, 2)
    get_left_child(config.tree, stack_word_index, child_number=2) |> word_index -> set_word_data_by_index_with_offset(word_index, 3)
    get_right_child(config.tree, stack_word_index, child_number=2) |> word_index -> set_word_data_by_index_with_offset(word_index, 4)
    get_left_child(config.tree, stack_word_index) |> 
      word_index -> get_left_child(config.tree, word_index) |>
      word_index -> set_word_data_by_index_with_offset(word_index, 5)
    get_right_child(config.tree, stack_word_index) |> 
      word_index -> get_right_child(config.tree, word_index) |>
      word_index -> set_word_data_by_index_with_offset(word_index, 6)
  end

  batch
end

function predict_transition(parser::DepParser, config::Configuration)
  form_batch(parser, config) |> 
    batch -> predict(parser.model, batch) |>
    findmax |>
    max_score_wiht_index -> parser.parsing_system.transitions[max_score_wiht_index[end]]
end

function get_word_id(parser::DepParser, word::String)
  haskey(parser.model.word_ids, word) ? parser.model.word_ids[word] : parser.model.word_ids[UNKNOWN_TOKEN]
end

function get_tag_id(parser::DepParser, tag::String)
  haskey(parser.model.tag_ids, tag) ? parser.model.tag_ids[tag] : parser.model.tag_ids[UNKNOWN_TOKEN]
end

function get_label_id(parser::DepParser, label::String)
  parser.model.label_ids[label]
end