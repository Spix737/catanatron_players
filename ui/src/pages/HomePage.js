import React, { useState } from "react";
import { useHistory } from "react-router-dom";

import "./HomePage.scss";
import { Button } from "@material-ui/core";
import Loader from "react-loader-spinner";
import { createGame } from "../utils/apiClient";

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const history = useHistory();

  const handleCreateGame = async (players) => {
    setLoading(true);
    const gameId = await createGame(players);
    setLoading(false);
    history.push("/games/" + gameId);
  };

  return (
    <div className="home-page">
      <h1>Settlers of Catan</h1>
      <h1>Deep Q-Network</h1>
      <ul>
          <li>1V1</li>
          <li>OPEN HAND</li>
          <li>NO CHOICE DURING DISCARD</li>
      </ul>
      <br/>
      <div className="switchable">
        {!loading && (
          <>
            <Button
              variant="contained"
              color="secondary"
              onClick={() => handleCreateGame(["DQN", "DQN"])}
            >
              Watch DQN 1 v 1
            </Button>
            <Button
              variant="contained"
              color="secondary"
              onClick={() => handleCreateGame(["DQN", "DQNS"])}
            >
              Watch 4p DQN FFA
            </Button>
            <Button
              variant="contained"
              color="tertiary"
              onClick={() => handleCreateGame(["HUMAN", "CATANATRON"])}
            >
              Play against Catanatron
            </Button>
            {/* <Button
              variant="contained"
              color="secondary"
              onClick={() => handleCreateGame(["RANDOM", "RANDOM"])}
            >
              Watch Random Bot 1 v 1
            </Button> */}
            <Button
              variant="contained"
              color="primary"
              onClick={() => handleCreateGame(["RANDOM", "RANDOMS"])}
            >
              Watch 4p Random Bots FFA
            </Button>

            {/* <Button
              variant="contained"
              color="primary"
              onClick={() => handleCreateGame(["CATANATRON", "CATANATRON"])}
            >
              Watch Catanatron 1 v 1
            </Button> */}
            <Button
              variant="contained"
              color="primary"
              onClick={() => handleCreateGame(["CATANATRON", "CATANATRONS"])}
            >
              Watch 4p Catanatron FFA
            </Button>
          </>
        )}
        {loading && (
          <Loader
            className="loader"
            type="Grid"
            color="#ffffff"
            height={60}
            width={60}
          />
        )}
      </div>
    </div>
  );
}
