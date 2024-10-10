import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QInputDialog, QScrollArea, QFrame)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Verificação e Identificação Facial')
        self.setGeometry(100, 100, 800, 600)  # Tamanho ajustado para comportar a área de pessoas cadastradas

        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # Criar o reconhecedor
        self.known_face_encodings = []  # Armazenar embeddings de rostos conhecidos
        self.known_face_names = []  # Armazenar nomes das pessoas conhecidas
        self.known_face_images = []  # Armazenar caminhos das imagens conhecidas

        # Layout principal
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Botão para carregar rosto conhecido
        self.load_known_button = QPushButton('Carregar Rosto Conhecido', self)
        self.load_known_button.clicked.connect(self.load_known_face)
        self.main_layout.addWidget(self.load_known_button)

        # Botão para carregar imagem para verificação/identificação
        self.load_image_button = QPushButton('Carregar Imagem para Identificação', self)
        self.load_image_button.clicked.connect(self.open_image)
        self.main_layout.addWidget(self.load_image_button)

        # Label para exibir a imagem
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.image_label)

        # Label para exibir o nome identificado
        self.name_label = QLabel(self)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.name_label)

        # Label para exibir detalhes adicionais
        self.details_label = QLabel(self)
        self.details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.details_label)

        # Área de rolagem para exibir as pessoas cadastradas
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        self.main_layout.addWidget(self.scroll_area)  # Adicionando a área de rolagem ao layout principal

        placeholder_pixmap = QPixmap(400, 300)
        placeholder_pixmap.fill(Qt.gray)  # Cor cinza como fundo
        self.image_label.setPixmap(placeholder_pixmap)

    def load_known_face(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, 'Escolha uma ou mais imagens de um rosto conhecido', '',
                                                'Images (*.png *.xpm *.jpg *.bmp *.gif *.jpeg)', options=options)
        if files:
            # Solicitar o nome da pessoa associada a essas imagens
            name, ok = QInputDialog.getText(self, 'Informe o nome', 'Nome do rosto conhecido:')
            if ok and name:
                # Verificar se o nome já foi cadastrado
                if name in self.known_face_names:
                    self.name_label.setText(f"O nome '{name}' já está cadastrado.")
                else:
                    for file_name in files:
                        img = cv2.imread(file_name)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # Adicionar o encoding à lista, associando ao mesmo nome
                        self.known_face_encodings.append(gray)
                        self.known_face_names.append(name)
                        self.known_face_images.append(file_name)  # Adicionar a imagem ao armazenamento

                    # Treinar o reconhecedor com as imagens conhecidas
                    self.face_recognizer.train(self.known_face_encodings, np.array(range(len(self.known_face_names))))

                    # Mostrar uma das imagens na interface e confirmar o cadastro
                    self.image_label.setPixmap(
                        QPixmap(files[0]).scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation))
                    self.name_label.setText(f"Rosto '{name}' cadastrado com sucesso. ({len(files)} imagens adicionadas)")

                    # Adicionar apenas a nova pessoa à lista de pessoas cadastradas
                    self.add_person_to_list(name, self.known_face_images[-1])

    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Escolha uma imagem para identificação', '',
                                                   'Images (*.png *.xpm *.jpg *.bmp *.gif *.jpeg)', options=options)
        if file_name:
            img = cv2.imread(file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Mostrar mensagem de progresso
            self.name_label.setText("Processando...")

            # Lógica de identificação
            label, confidence = self.face_recognizer.predict(gray)
            details = f"Confiança: {confidence:.2f}\n"

            if confidence < 100:  # Valor de confiança
                name = self.known_face_names[label]
                self.name_label.setText(f"Identificado: {name} (Confiança: {confidence:.2f})")
                details += f"Pessoa identificada como: {name}\n"
                details += "Rosto mais próximo encontrado nos conhecidos.\n"
            else:
                self.name_label.setText("Rosto desconhecido.")
                details += "Rosto não identificado com confiança suficiente.\n"

            # Exibir detalhes adicionais da comparação
            self.details_label.setText(details)

            # Mostrar a imagem na interface
            self.image_label.setPixmap(
                QPixmap(file_name).scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))

    def add_person_to_list(self, name, image_path):
        # Criar um layout horizontal para adicionar a nova pessoa
        person_layout = QHBoxLayout()

        # Exibir a foto da pessoa
        pixmap = QPixmap(image_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)
        image_label = QLabel(self)
        image_label.setPixmap(pixmap)

        # Exibir o nome da pessoa
        name_label = QLabel(name, self)

        # Adicionar ao layout horizontal
        person_layout.addWidget(image_label)
        person_layout.addWidget(name_label)

        # Adicionar o layout da nova pessoa ao layout da área de rolagem
        self.scroll_layout.addLayout(person_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
