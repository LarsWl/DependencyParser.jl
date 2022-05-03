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
    threads_count::Integer


    TrainingContext(
        system::ParsingSystem,
        settings::Settings,
        connlu_sentences::Vector{ConnluSentence},
        test_connlu_sentences::Vector{ConnluSentence},
        test_connlu_file::String,
        training_results_file::String,
        model_file::String,
        threads_count = 6
    ) = new(system, settings, connlu_sentences, test_connlu_sentences, test_connlu_file, training_results_file, model_file, undef, 0, 0, Inf, threads_count)
end