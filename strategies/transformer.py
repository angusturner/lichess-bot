import functools
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from chess import Board
from chess.engine import PlayResult

from neural_chess.utils.data import board_to_flat_repr, get_legal_move_mask, sample_move
from .minimal_engine import MinimalEngine
from hijax.setup import setup_worker

from neural_chess import MODULE_NAME

# force CPU
jax.config.update("jax_platform_name", "cpu")


class TransformerEngine(MinimalEngine):
    def __init__(
        self,
        _commands,
        _options,
        _stderr,
        experiment_name: str = "bot_large",
        checkpoint_id: str = "best",
        exp_dir: Optional[str] = None,
        target_elo: int = 2500,
    ):
        super(TransformerEngine, self).__init__(name="transformer")
        self.target_elo = target_elo

        # load the worker
        worker, cfg = setup_worker(
            name=experiment_name,
            module=MODULE_NAME,
            with_wandb=False,
            with_loaders=False,
            checkpoint_id=checkpoint_id,
            exp_dir=exp_dir,
        )
        self.worker = worker
        self.params = worker.params
        self.rng_key = worker.rng_key

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_move_probs(self, board_state, turn, castling_rights, en_passant, elo, legal_moves, **_kwargs):
        logits = self.worker.forward.apply(
            self.params, self.rng_key, board_state, turn, castling_rights, en_passant, elo, is_training=False
        )
        logits = jnp.where(legal_moves, logits, jnp.full_like(logits, -1e9))
        return jax.nn.softmax(logits, axis=-1)

    def search(self, board: Board, _timeleft: Any, ponder: bool, draw_offered: bool) -> PlayResult:
        """
        :param board:
        :param _timeleft:
        :param ponder:
        :param draw_offered:
        :return:
        """
        # encode the board
        board_state = board_to_flat_repr(board).astype(np.int32)

        # get the turn, castling rights, etc...
        turn = board.turn
        castling_rights = board.has_castling_rights(turn)
        elo = self.target_elo / 2500  # approx. in [0, 1]

        # is there an en-passant square?
        # - [0, 63] indicating the position that can be moved to with en-passant
        # - 64 indicating no en-passant rights
        en_passant = board.ep_square if board.ep_square else 64

        # legal moves mask!
        legal_moves = get_legal_move_mask(board).astype(bool)

        # convert stuff to arrays with batch dimension
        batch = {
            "board_state": board_state.reshape([1, -1]),
            "turn": np.array([board.turn]).astype(np.int32),
            "elo": np.array([elo]).astype(np.float32),
            "en_passant": np.array([en_passant]).astype(np.int32),
            "castling_rights": np.array([castling_rights]).astype(np.int32),
            "legal_moves": legal_moves.reshape([1, -1]),
        }

        move_probs = np.array(self._get_move_probs(**batch))[0]
        next_move, _ = sample_move(move_probs, greedy=True)
        return PlayResult(next_move, None)
