import Navbar from '../../components/Navbar/Navbar';
import './appLayout.css';

function AppLayout({ children }) {
  return (
    <>
      <Navbar />
      <main className="app-main">{children}</main>
    </>
  );
}

export default AppLayout;
