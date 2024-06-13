import re

def limpiar_texto(file_path, output_path):
    # Expresión regular para encontrar y eliminar "training/" seguido de un número del 1 al 14818
    pattern = re.compile(r'training/')
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Reemplazar las coincidencias con una cadena vacía
    cleaned_lines = [pattern.sub('', line) for line in lines]

    with open(output_path, 'w') as output_file:
        output_file.writelines(cleaned_lines)

# Ruta del archivo de entrada y salida
input_file = 'reuters/cats.txt'
output_file = 'cats.txt'

# Llamar a la función para limpiar el texto
limpiar_texto(input_file, output_file)
