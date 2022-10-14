# Cohere Learn

Cohere learn is a set of utilitites which wrap around the Cohere Python API for common tasks. It also builds on common data formats e.g. Pandas and Huggingface Datasets (coming soon).

Here are some tasks of interest:

- [x] Few Shot, Multi-Class, Single Label Classification, uses: Classify
- [ ] Few Shot, Multi-class, _Multi-label_ classification, uses: Classify
- [ ] Keywords Extraction with a biased prompt, uses: Generate

## Few Shot, Mult-Class, Single Label Classification

Inititalising the pipeline:

```python
cohere_clf = FewShotClassify(
    df=df,
    cohere_client=co,
    train_counts=[4, 8, 16, 32],
    test_count=64,
    x_label="Name",
    y_label="Key",
)
```

The `predict` function is what makes the web calls. The responses are not stored in the object, but the train and test splits are.

```python
responses = cohere_clf.predict()
```

These responses are of the type which Cohere Python SDK returns, and can be iterated over to get the results.

```bash
>> r = responses[0]
>> type(r), type(r.classifications), type(r.classifications[0])
(cohere.classify.Classifications, list, cohere.classify.Classification)
```
