import { useEffect, useRef } from 'react';
import './modal.css';

function Modal({ title, isOpened, onClose, children }) {
  const dialogRef = useRef();
  useEffect(() => {
    if (isOpened) {
      dialogRef.current?.showModal();
    } else {
      dialogRef.current?.close();
    }
  }, [isOpened]);

  const onBackgroundClick = (event) => {
    if (event.target.tagName === 'DIALOG') {
      onClose(event);
    }
  };

  const onEscPress = (event) => {
    if (event.key === 'Escape') {
      onClose(event);
    }
  };

  return (
    // onKeyDownCapture is not caught by this rule for some reason.
    // eslint-disable-next-line jsx-a11y/click-events-have-key-events
    <dialog
      ref={dialogRef}
      className="modal-container-wrapper"
      onClick={onBackgroundClick}
      onKeyDownCapture={onEscPress}
      tabIndex="0"
    >
      <div className="modal-topbar-container">
        <button className="modal-close-button" onClick={onClose}>
          <img
            className="modal-close-button__icon"
            src="close.svg"
            alt="close"
          />
        </button>
      </div>

      <p className="modal-title main-font">{title}</p>
      <div className="modal-content">{children}</div>
    </dialog>
  );
}

export default Modal;
