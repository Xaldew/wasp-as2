from convnets import *


def main(output_dir, enum, final, n_unique_words, max_review_length, dropout):
    """Evaluate the convolutional model.

    """

    (_, _), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words)
    pad_type = trunc_type = 'pre'
    x_valid = pad_sequences(x_valid, maxlen=max_review_length,
                            padding=pad_type, truncating=trunc_type, value=0)

    model = get_model(enum, n_unique_words, max_review_length, dropout)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Load previously prepared weights.
    output_fmt = os.path.join(output_dir, "weights.{:02d}.hdf5".format(final))
    model.load_weights(output_fmt)

    # use the model to estimate the validation set.
    y_hat = model.predict_proba(x_valid)
    print("{:0.2f}".format(roc_auc_score(y_valid, y_hat)*100.0))

    plt.hist(y_hat)
    _ = plt.axvline(x=0.5, color='orange')
    plt.show()

    return 0


if __name__ == "__main__":
    ARGS = parse_arguments(sys.argv[1:])
    locale.setlocale(locale.LC_ALL, "")
    random.seed(ARGS.seed)
    sys.exit(main(ARGS.output_dir,
                  ARGS.model,
                  ARGS.final,
                  ARGS.unique_words,
                  ARGS.max_review_length,
                  ARGS.dropout))
