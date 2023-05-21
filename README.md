**Word Probing**

This is a repository implementing word probing experiments. To learn more about what probing is, read [this](https://arxiv.org/abs/2102.12452) paper. I used sentences in Universal Dependencies format to extract data to conduct probing experiments on. This projects focuses on word probing, that is, probing features that refer to a word (or several words), not a sentence. Some examples could be gender of a word, or predicting if one word depends on the other in the sentence from which they are taken.

In this project, two types of word probing experiments are implemented and and compared. The first type is contextualized word probing, the second one is non-contextualized word probing. The first type means that we put a whole sentence into a language model, seek out the embeddings that correspond to a word that is interesting for us, and use those embeddings as features in the probing experiments. The second type means simply putting a word into a language model, get the embeddings for the word as if it were a whole sentence and use those embeddings for probing.

The purpose of this project was to compare the two approaches and hopefully show that the first one produces better results not only on context-related features, but also on simple features like gender or number.

The visualization for each experiment consists of two boxplots. The first boxplot shows the distribution of the quality of the logistic regression trained on the embeddings from each layer of the language model, the second boxplot shows the distribution of the number of iterations required for the regression to converge, also for every layer of the language model.

In this project we chose to work with the French language as it is rich with linguistic features (when compared to English). We are working with Camembert model trained on the Frnech laguage.

__Warning__: if you run this in google colab, the multithreading does not speed things up, so the heaviest experiments may run for a long time (an hour maybe? I haven't tried.) (see [this](https://stackoverflow.com/questions/76048467/stop-numpysklearn-from-multithreading-in-google-colab)).

**Documentation**

`get_average_embs(sentence, layer_inds, model=model, tokenizer=tokenizer, device=DEVICE)` is the function that accepts a sentence, a list of indices of layers for the experiment, and also information about the language model that you generally don't need to change.

It returns a two-dimensional torch stack that contains an embedding vector for the sentence for each layer of the model.

In this project, the function is used to calculate non-contextual word embeddings.

`TokenMerger` - this is an interface used by get_word_embs function to merge the tokens into words and optionally calculate their embeddings. This class is needed to be able to replace Camembert model with another model (maybe for another language). Every model uses a different notation to mark the beginning of a new word, thus we need this interface provided along with the model. It needs to provide an `__init__(self, calculateEmbeddings=False)` method. The calculateEmbeddings parameter indicates if this instance of the class will need to calculate embeddings. The finish method should return a tuple containing a list of resulting tokens and a list of their embeddings if calculateEmbeddings is True and just the tokens otherwise. The `next_token(self, token, embedding=None)` method accepts the next token (a string) and its embedding if calculateEmbeddings is True. It does not return anything, but updates the class, adding the token and its embedding.

`get_word_embs(sentence, word_inds, layer_inds, TokenMerger, model=model, tokenizer=tokenizer, device=DEVICE)` - this function accepts the sentence and the indices of the words for which we need to extract the embeddings, also the layer indices and the aforementioned TokenMerger class and the information about the model. It returns a three-dimensional torch stack, the first dimension corresponds to a layer, the second to a word, the third is the embedding vector.

`join_parsings(bert_tokenization, ud_tokenization, TokenMerger, verbose=False)` - this function accepts two tokenization, the model token list and the ud tokenization. It also accepts a TokenMerger to work with the first tokenization. First it uses TokenMerger to compile the first token list into a word list. Then it compares the two tokenizations (namely, it merges some of the ud tokens and their features into one to match the model tokenization). The function returns three lists, the list of words, the list of ud features for each word and the dict mapping every internal ud index into the index of that word in the resulting word tokenization (the latter two objects are required in the functions that extract the features, see features_extractor below).

This function is required to mitigate the effects of the two tokenizations not matching. An empirical observation is that in most of cases where they mismatch one token in the model tokenization corresponds to more than one token in the ud tokenization (for example, "a-t-il" is a single token in Camembert but two tokens in ud). We do not need to cover all of the cases, but we want to maximise the number of data we get, so for now we chose this strategy to merge the tokenization.

`features_extractor(feats, orig_index)` is a function that accepts the list of features and the dictionary mentioned in the join_parsings section above. It returns a list, every item of which is a single observation for the dataset, and a list of target variables for those observations. An item is represented by a list of indices of words that partake in this experiment (For a simple experiment like number prediction this would be one word, but for an experiment like dependency prediction it can be two words. Theoretically it can be more than two words, it will still work.).

When you are defining a new experiment, this function is what you will need to define. If it is a simple experiment predicting a feature for a subset of part of speech types, you can simply construct the function with the `create_word_by_word_parser([POS_1, POS_2, ... POS_n], "<Feature>")`.

`load_word_in_sentence_data(dataset_name, TokenMerger, features_extractor, verbose=False)` - this function creates a dataset for contextualized experiments from a local file, accepting the file name, TokenMerger, and a features_extractor function. The dataset has three columns "text" - a sentence, "word_ind" - list of lists of word indices for every observation extracted from this sentence and "target" - list of target variables for those observations. The observation are grouped by the sentence so that the experiments can every sentence through the language model only one time.

`load_separate_word_data(dataset_name, TokenMerger, features_extractor, verbose=False)` - this function creates a dataset for non-contextualized experiments from a local file, accepting the file name, TokenMerger and a features_extractor function The dataset has two columns: "word" and "target", each row is a single observation (maybe consisting of several words). There is no need to group observations by sentence in this case.

You may wonder why we need a TokenMerger for non-contextualized experiments. This is so that the experiments of different types are comparable, that is, they work with exactly the same data. The join_parsings function will fail on the same sentences as everything is determinate.

`rerun_logregs(data, targets, runs, threads)` is a function that reruns the logistic regressions the amount of times determined in the runs argument, using several threads. It returns the scores and iterations to convergence grouped by the layer.

`word_in_sentence_experiment(layers, TokenMerger, data_extractor, verbose=False, runs=1, threads=1)` is a function that runs the contextualized experiment, using all the functions above and returns the result of rerun_logregs.

`separate_word_experiment(layers, TokenMerger, data_extractor, verbose=False, runs=1, threads=1)` is a function that runs the non-contextualized experiment, the rest is the same as above.

`show_results(quals, iternums, title)` is a helper function that visualizes the results of experiments in two boxplots.