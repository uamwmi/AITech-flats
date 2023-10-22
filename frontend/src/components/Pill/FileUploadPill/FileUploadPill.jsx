import Pill from '../Pill';
import './fileUploadPill.css';

function FileUploadPill({ text, id, onChange }) {
  return (
    <Pill type="button" className="file-upload-wrapper">
      <label className="fileupload-input-container" htmlFor={id}>
        <span className="fileupload-input-label secondary-font">
          {text || 'Wybierz plik'}
        </span>
        <img className="fileupload-input-icon" alt="upload" src="upload.svg" />
        <input
          accept="image/png, image/gif, image/jpeg"
          className="fileupload-input-field"
          id={id}
          type="file"
          name="filename"
          onChange={onChange}
        />
      </label>
    </Pill>
  );
}

export default FileUploadPill;
