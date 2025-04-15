from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit


class HelpMenu(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Help")

        self.resize(800, 600)

        # Create a layout for the dialog
        layout = QVBoxLayout()

        # Add a text area for help content
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setPlainText("This application offers the following keyboard shortcuts:\n\n"
                               "Play / Pause: Spacebar\n"
                               "Speed increase forward: Arrow Key Right →\n"
                               "Speed increase backwards: Arrow Key Right ←\n"
                               "Select next Elephant name: Arrow Key Down ↓\n"
                               "Select previous Elephant name: Arrow Key Up ↑\n")

        layout.addWidget(help_text)

        self.setLayout(layout)
