import { Link } from 'react-router-dom';
import AppLayout from '../layouts/AppLayout/AppLayout';

function HomePage() {
  return (
    <AppLayout>
      <Link to="tell">
        <h1 className="main-font">Tell</h1>
      </Link>
      <Link to="shift">
        <h1 className="main-font">Shift</h1>
      </Link>
    </AppLayout>
  );
}

export default HomePage;
