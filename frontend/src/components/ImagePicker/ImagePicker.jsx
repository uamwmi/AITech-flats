import { useState } from 'react';
import Modal from '../../layouts/Modal/Modal';
import './imagePicker.css';
import PillButton from '../Pill/PillButton';
import { getLocalRequest } from '../../api/request';

function ImagePicker({ isOpened, onClose, onImageSelected }) {
  const imageNames = ['1', '2', '3', '4', '5', '6'];
  const format = '.webp';
  const [pickedItem, setPickedItem] = useState(null);
  const imagePath = (name) => `examples/${name}${format}`;
  const onDone = async () => {
    const localImageRequest = await getLocalRequest(imagePath(pickedItem));
    const blob = await localImageRequest.blob();
    onImageSelected({
      name: imagePath(pickedItem),
      data: blob,
    });
    onClose();
  };
  const onImageClick = (imageName) => {
    setPickedItem(imageName);
  };
  return (
    <Modal isOpened={isOpened} onClose={onClose} title="Wybierz obraz">
      <div className="image-list-container">
        {imageNames.map((imageName) => (
          // eslint-disable-next-line jsx-a11y/click-events-have-key-events
          <img
            key={imageName}
            className={`image-list-item ${
              imageName === pickedItem ? 'image-list-item__selected' : ''
            }`}
            src={`${imagePath(imageName)}`}
            alt={`example_image_${imageName}`}
            onClick={() => onImageClick(imageName)}
          />
        ))}
      </div>
      <div className="image-picker-buttons-container">
        <PillButton onClick={onClose}>Wróć</PillButton>
        <PillButton onClick={onDone} selected inactive={pickedItem === null}>
          Wybierz
        </PillButton>
      </div>
    </Modal>
  );
}

export default ImagePicker;
