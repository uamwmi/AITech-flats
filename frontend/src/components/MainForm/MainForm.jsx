import { useState } from 'react';
import ListWithHeading from '../../layouts/ListWithHeading/ListWithHeading';
import PillButton from '../Pill/PillButton';
import FileUploadPill from '../Pill/FileUploadPill/FileUploadPill';
import './mainForm.css';
import ImagePicker from '../ImagePicker/ImagePicker';

function MainForm({ children, onSubmit }) {
  const [file, setFile] = useState({ name: '', data: null });
  const [modalActive, setModalActive] = useState(false);
  const onFileChange = (event) => {
    if (event.target.files.length < 1) {
      setFile({ data: null, name: '' });
      return;
    }
    setFile({
      data: event.target.files[0],
      name: event.target.files[0].name,
    });
  };

  const onExamplesModalOpen = () => {
    setModalActive((prev) => !prev);
  };
  return (
    <>
      <form className="filling-layout-container">
        <ListWithHeading headerText="Zdjęcie">
          <FileUploadPill
            id="main-image"
            onChange={onFileChange}
            text={file.name}
          />
          <span className="secondary-font">Lub</span>
          <PillButton onClick={onExamplesModalOpen}>
            Wybierz z przykładów
          </PillButton>
        </ListWithHeading>
        {children}

        <div className="form-centered-button">
          <PillButton
            inactive={!file.name}
            selected
            onClick={() => onSubmit(file)}
          >
            Jazda
          </PillButton>
        </div>
      </form>
      <ImagePicker
        isOpened={modalActive}
        onClose={() => setModalActive(false)}
        onImageSelected={(selectedFile) => {
          console.log(selectedFile);
          setFile(selectedFile);
        }}
      />
    </>
  );
}

export default MainForm;
