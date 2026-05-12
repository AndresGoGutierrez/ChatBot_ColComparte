from src.retriever import retrieve_context

def main():
    query = "¿Quiénes fundaron Colombia Comparte?"
    r = retrieve_context(query)

    for x in r["results"]:
        print(f"score={x['score']} | {x['source']}")
        print(x["text"][:200])
        print()


if __name__ == "__main__":
    main()