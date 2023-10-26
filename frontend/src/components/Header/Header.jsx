import './header.css';

function Header({ heading, subheading }) {
  return (
    <article className="header-container">
      <h1 className="header-heading main-font">{heading}</h1>
      <p className="header-subheading secondary-font">{subheading}</p>
    </article>
  );
}

export default Header;
