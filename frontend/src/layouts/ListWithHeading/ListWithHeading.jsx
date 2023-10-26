import './listWithHeading.css';
import '../FillingLayout/fillingLayout.css';

function ListWithHeading({ headerText, children }) {
  return (
    <div className="headed-list-container filling-layout-container">
      <p className="headed-list-paragraph secondary-font">{headerText}</p>
      <div className="headed-list-items">{children}</div>
    </div>
  );
}

export default ListWithHeading;
