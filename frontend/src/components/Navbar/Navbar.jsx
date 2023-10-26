import './navbar.css';
import { Link } from 'react-router-dom';

function Navbar() {
  return (
    <nav className="main-navbar-container">
      <Link to="/" className="main-navbar-header__link">
        <h2 className="main-navbar-header main-font">Inter.io</h2>
      </Link>
      <img className="main-navbar-kebab" src="Kebab.svg" alt="Kebab" />
    </nav>
  );
}

export default Navbar;
