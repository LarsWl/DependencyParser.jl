mutable struct TrainingContext
    system::ParsingSystem
    settings::Settings
    connlu_sentences::Vector{ConnluSentence}
    test_connlu_sentences::Vector{ConnluSentence}
    test_connlu_file::String
    training_results_file::String
    model_file::String
    optimizer
    best_uas
    best_las
    best_loss


    TrainingContext(
        system::ParsingSystem,
        settings::Settings,
        connlu_sentences::Vector{ConnluSentence},
        test_connlu_sentences::Vector{ConnluSentence},
        test_connlu_file::String,
        training_results_file::String,
        model_file::String
    ) = new(system, settings, connlu_sentences, test_connlu_sentences, test_connlu_file, training_results_file, model_file, undef, 0, 0, Inf)
end