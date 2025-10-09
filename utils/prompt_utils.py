# utils/prompt_utils.py
def generate_prompt_from_summary(genre, subfield, year, keywords, summary, word_count):
    keywords_line = ", ".join(keywords)

    base = f"""Here is a summary of a human-written {genre.lower()} text:

Keywords: {keywords_line}
Summary: {summary}
"""

    if genre == "Academic":
        instruction = f"""Now, write a new academic-style abstract inspired by these ideas.
Do not copy or paraphrase the original text — write an original, conceptually similar discussion in {subfield}.
Use a formal academic tone (~{word_count} words). Year context: {year}."""
    elif genre == "Blogs":
        instruction = f"""Now, write a new blog post inspired by these ideas for {subfield}.
Use a conversational tone (~{word_count} words). Year context: {year}."""
    elif genre == "News":
        instruction = f"""Now, write a new news article inspired by these ideas for {subfield}.
Keep journalistic neutrality (~{word_count} words). Year context: {year}."""
    else:
        raise ValueError(f"Unsupported genre: {genre}")

    return (base + "\n" + instruction).strip()
