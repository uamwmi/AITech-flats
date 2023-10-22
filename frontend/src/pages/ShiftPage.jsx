import AppLayout from '../layouts/AppLayout/AppLayout';
import Header from '../components/Header/Header';
import PillButton from '../components/Pill/PillButton';
import MainForm from '../components/MainForm/MainForm';
import ListWithHeading from '../layouts/ListWithHeading/ListWithHeading';

function ShiftPage() {
  return (
    <AppLayout>
      <Header
        heading="INTER.IO/shift"
        subheading="Zobacz swoje mieszkanie w innej odsłonie!"
      />
      <MainForm onSubmit={() => {}}>
        <ListWithHeading headerText="Styl">
          <PillButton onClick={() => {}}>Skandynawski</PillButton>
          <PillButton onClick={() => {}} selected>
            Komuna
          </PillButton>
          <PillButton onClick={() => {}}>Nowoczesny</PillButton>
          <PillButton onClick={() => {}}>Boho</PillButton>
          <PillButton onClick={() => {}}>Rustykalny</PillButton>
        </ListWithHeading>
      </MainForm>
    </AppLayout>
  );
}

export default ShiftPage;
