import wikipediaapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
import json


def get_game_of_thrones_content():
    wiki = wikipediaapi.Wikipedia(
        user_agent='GameOfThronesTriviaBot/1.0',
        language='en'
    )
    page = wiki.page('Game_of_Thrones')

    sections = ['Plot', 'Episodes', 'Main_characters']
    content = page.summary
    print(f"Extracted summary: {len(page.summary)} characters")

    for section in sections:
        if section in page.sections:
            section_content = page.sections[section].text
            content += f"\n\n{section}:\n{section_content}"
            print(f"Added '{section}' section: {len(section_content)} characters")

    print(f"Total content extracted: {len(content)} characters")
    return content


def create_test_questions():
    test_questions = [
        {
            "question": "Who are the main characters in Game of Thrones?",
            "reference": "The main characters include members of several noble houses: the Starks (Eddard, Catelyn, Robb, Sansa, Arya, Bran, Rickon), the Lannisters (Tywin, Cersei, Jaime, Tyrion), Daenerys Targaryen, and Jon Snow."
        },
        {
            "question": "What is the Iron Throne?",
            "reference": "The Iron Throne is the throne of the Seven Kingdoms, forged by Aegon the Conqueror from the swords of his defeated enemies. It is located in the Red Keep in King's Landing and is the ultimate symbol of power in Westeros."
        },
        {
            "question": "What are the main plot points of Game of Thrones?",
            "reference": "The main plot follows several noble houses vying for control of the Iron Throne, while an ancient enemy (the White Walkers) threatens from the North. Key events include Ned Stark's execution, the War of the Five Kings, Daenerys's rise in Essos, and the eventual battle against the Night King."
        }
    ]
    with open("test_questions.json", "w") as f:
        json.dump(test_questions, f)
    return test_questions


def create_vector_db():
    got_content = get_game_of_thrones_content()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(got_content)
    print(f"Split content into {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model="llama3.2:latest")

    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./got_db_llama3.2"
    )
    vector_db.persist()
    print("Vector DB created successfully with llama3.2!")

    # Create test questions
    create_test_questions()
    print("Test questions created in test_questions.json")


if __name__ == "__main__":
    create_vector_db()