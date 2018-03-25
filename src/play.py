import game
import numpy as np
from learning import neural_net
SENSORS = 3
def play(model):

    score = 0
    game_state = game.GameState()
    _, state = game_state.frame_step((2))

    while True:
        score += 1

        action = (np.argmax(model.predict(state, batch_size=1)))

        _, state = game_state.frame_step(action)

        if score % 1000 == 0:
            print("Current Score : %d frames." % score)

if __name__ == "__main__":
    saved_model = 'models/164-150-400-50000-100000.h5'
    model = neural_net(SENSORS, [164, 150], saved_model)
    play(model)
