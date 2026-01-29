from canapy import metrics
from canapy.annotator import SynAnnotator

# def test_sklearn_confusion_matrix(corpus):
    
#     annotated_corpus = 
#     print(metrics.sklearn_confusion_matrix(corpus, corpus))

# def test_sklearn_classification_report(corpus):
#     print(metrics.sklearn_classification_report(corpus, corpus))
    
def test_segment_error_rate(real_annot_corpus, real_nonannot_corpus, spec_directory):
    annotator = SynAnnotator(
        config=real_annot_corpus.config,
        spec_directory=spec_directory,
    )

    annotator.fit(real_annot_corpus)

    pred_corpus = annotator.predict(real_annot_corpus)
    print(metrics.segment_error_rate(real_annot_corpus, pred_corpus))

