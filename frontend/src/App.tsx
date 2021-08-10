import { makeStyles } from "@material-ui/core";
import Container from "@material-ui/core/Container";
import FilledInput from "@material-ui/core/FilledInput";
import FormControl from "@material-ui/core/FormControl";
import Grid from "@material-ui/core/Grid";
import InputAdornment from "@material-ui/core/InputAdornment";
import InputLabel from "@material-ui/core/InputLabel";
import React from "react";

function App() {
  const [input, setInput] = React.useState("");

  const useStyles = makeStyles({
    root: {
      flexGrow: 1,
    },
    search: {
      marginTop: 250,
      width: 200,
    },
  });

  const handleChange = () => (event: React.ChangeEvent<HTMLInputElement>) => {
    setInput(event.target.value);
  };

  const classes = useStyles();
  return (
    <div className={classes.root}>
      <Grid
        container
        direction="row"
        justifyContent="center"
        alignItems="center"
      >
        <div className={classes.search}>
          <FormControl variant="filled">
            <InputLabel htmlFor="filled-adornment-amount">Url Input</InputLabel>
            <FilledInput
              id="filled-adornment-amount"
              value={input}
              onChange={handleChange()}
              startAdornment={
                <InputAdornment position="start">$</InputAdornment>
              }
            />
          </FormControl>
        </div>
      </Grid>
    </div>
  );
}

export default App;
