if __name__ == '__main__':
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-mpnet-base-v2")
    query_embedding = model.encode("How big is London")
    passage_embedding = model.encode([
        "London has 9,787,426 inhabitants at the 2011 census",
        "London is known for its finacial district",
    ])

    print("Similarity:", util.dot_score(query_embedding, passage_embedding))
