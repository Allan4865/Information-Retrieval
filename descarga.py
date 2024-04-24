import requests
import os

def download_book(book_id, folder_path):
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder_path, f"pg{book_id}.txt"), 'wb') as file:
            file.write(response.content)
        print(f"Libro {book_id} descargado con éxito.")
    else:
        print(f"No se pudo descargar el libro {book_id}.")

def main():
    # Carpeta donde se guardarán los libros
    folder_path = "/data folder"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Archivo que contiene los números de los libros
    numbers_file = "numeros.txt"
    
    # Descargar los libros
    with open(numbers_file, 'r') as file:
        for line in file:
            book_id = line.strip()
            download_book(book_id, folder_path)

if __name__ == "__main__":
    main()
