package dk.easv.bll.bot;

import dk.easv.bll.field.IField;
import dk.easv.bll.game.GameState;
import dk.easv.bll.game.IGameState;
import dk.easv.bll.move.IMove;
import dk.easv.bll.move.Move;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class EmilBot implements IBot {

    private static final int    TIME_BUFFER_MS = 50;
    private static final double UCT_C          = 1.41;
    /**
     * RAVE equivalence parameter.
     * At visits = RAVE_K/3 the MCTS and RAVE estimates are weighted equally.
     * Tune between 100–2000; 500 is a good default for UTTT.
     */
    private static final double RAVE_K         = 500.0;

    private int botPlayer;

    // -----------------------------------------------------------------------
    // Data structures
    // -----------------------------------------------------------------------

    /** Carries the rollout outcome plus per-player move sets for RAVE updates. */
    private static class RolloutResult {
        /** Result from the leaf node's mover perspective: +1 win, -1 loss, 0 tie. */
        final int result;
        /** Absolute winner: 0, 1, or -1 for tie/draw. */
        final int winner;
        final Set<IMove> movesP0;
        final Set<IMove> movesP1;

        RolloutResult(int result, int winner, Set<IMove> movesP0, Set<IMove> movesP1) {
            this.result   = result;
            this.winner   = winner;
            this.movesP0  = movesP0;
            this.movesP1  = movesP1;
        }
    }

    private static class Node {
        Node         parent;
        List<Node>   children     = new ArrayList<>();
        List<IMove>  untriedMoves = new ArrayList<>();
        IMove        move;       // move that created this node
        int          mover;      // player who made that move
        IGameState   state;
        boolean      terminal;

        // MCTS statistics
        int    visits = 0;
        double wins   = 0;

        // RAVE / AMAF statistics  (same perspective as wins: the node's mover)
        int    raveVisits = 0;
        double raveWins   = 0;
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    @Override
    public IMove doMove(IGameState gameState) {
        long endTime = System.currentTimeMillis() + gameState.getTimePerMove() - TIME_BUFFER_MS;
        botPlayer = gameState.getMoveNumber() % 2;

        Node root = makeRoot(gameState);

        while (System.currentTimeMillis() < endTime) {
            Node leaf  = select(root);
            Node child = expand(leaf);
            RolloutResult rr = rollout(child);
            backpropagate(child, rr);
        }

        return bestMove(root, gameState);
    }

    @Override
    public String getBotName() {
        return getClass().getSimpleName();
    }

    // -----------------------------------------------------------------------
    // MCTS phases
    // -----------------------------------------------------------------------

    private Node select(Node node) {
        while (!node.terminal && node.untriedMoves.isEmpty() && !node.children.isEmpty()) {
            node = bestUCT(node);
        }
        return node;
    }

    /**
     * UCT selection blended with RAVE via the standard β formula:
     *   β  = sqrt( K / (3·visits + K) )
     *   score = (1−β)·Q_mcts + β·Q_rave + C·sqrt(ln(parent.visits) / visits)
     *
     * When visits are low β≈1 so RAVE dominates; as visits grow β→0 and pure
     * MCTS takes over.
     */
    private Node bestUCT(Node parent) {
        double logParent = Math.log(parent.visits + 1);
        return Collections.max(parent.children, Comparator.comparingDouble(n -> {
            double q     = n.visits     > 0 ? n.wins     / n.visits     : 0.0;
            double qRave = n.raveVisits > 0 ? n.raveWins / n.raveVisits : 0.0;
            double beta  = Math.sqrt(RAVE_K / (3.0 * n.visits + RAVE_K));
            double exploit = (1.0 - beta) * q + beta * qRave;
            double explore = UCT_C * Math.sqrt(logParent / (n.visits + 1e-9));
            return exploit + explore;
        }));
    }

    private Node expand(Node node) {
        if (node.terminal || node.untriedMoves.isEmpty()) {
            return node;
        }

        List<IMove> untried = node.untriedMoves;
        int idx  = ThreadLocalRandom.current().nextInt(untried.size());
        IMove mv = untried.remove(idx);

        GameSimulator sim = createSimulator(node.state);
        sim.updateGame(mv);

        Node child      = new Node();
        child.parent    = node;
        child.move      = mv;
        child.mover     = node.state.getMoveNumber() % 2;   // player who just moved
        child.state     = sim.getCurrentState();
        child.terminal  = (sim.gameOver != GameOverState.Active);

        if (!child.terminal) {
            List<IMove> avail = child.state.getField().getAvailableMoves();
            if (avail.isEmpty()) {
                child.terminal = true;
            } else {
                child.untriedMoves = new ArrayList<>(avail);
            }
        }

        node.children.add(child);
        return child;
    }

    /**
     * Plays out a random/heuristic game from {@code node} and records every
     * move made by each player so backprop can update RAVE counters.
     */
    private RolloutResult rollout(Node node) {
        Set<IMove> movesP0 = new HashSet<>();
        Set<IMove> movesP1 = new HashSet<>();

        // Terminal node: evaluate the macro board directly
        if (node.terminal) {
            int eval       = evalMacro(node.state.getField().getMacroboard(), node.mover);
            int absWinner  = (eval == 1) ? node.mover : (eval == -1 ? 1 - node.mover : -1);
            return new RolloutResult(eval, absWinner, movesP0, movesP1);
        }

        GameSimulator sim   = createSimulator(node.state);
        int           depth = 0;

        while (sim.gameOver == GameOverState.Active && depth < 200) {
            List<IMove> moves = sim.getCurrentState().getField().getAvailableMoves();
            if (moves.isEmpty()) break;

            IMove chosen = heuristicMove(sim, moves);
            // Track which player made which move for RAVE
            (sim.currentPlayer == 0 ? movesP0 : movesP1).add(chosen);
            sim.updateGame(chosen);
            depth++;
        }

        int result;
        int absWinner;
        if (sim.gameOver == GameOverState.Win) {
            // currentPlayer is about to move, so the winner is the previous player
            int winner = (sim.currentPlayer + 1) % 2;
            absWinner  = winner;
            result     = (winner == node.mover) ? 1 : -1;
        } else {
            absWinner = -1;
            result    = 0;
        }

        return new RolloutResult(result, absWinner, movesP0, movesP1);
    }

    /**
     * Standard backpropagation PLUS RAVE updates.
     *
     * For every node N on the path we check each already-expanded child C.
     * If C's move was played by the player-to-move at N during the simulation,
     * we credit C's RAVE counters with the result from that player's perspective.
     * This lets rarely-visited children benefit from statistics gathered in
     * other parts of the tree.
     */
    private void backpropagate(Node node, RolloutResult rr) {
        int result = rr.result;

        while (node != null) {
            node.visits++;
            node.wins += result;

            // ---- RAVE update for existing children -------------------------
            // toMove = player whose turn it is at this node (they will choose a child)
            int        toMove        = node.state.getMoveNumber() % 2;
            Set<IMove> relevantMoves = (toMove == 0) ? rr.movesP0 : rr.movesP1;

            // Result from toMove's perspective
            int raveResult;
            if      (rr.winner == toMove) raveResult =  1;
            else if (rr.winner == -1)     raveResult =  0;
            else                          raveResult = -1;

            for (Node child : node.children) {
                if (child.move != null && relevantMoves.contains(child.move)) {
                    child.raveVisits++;
                    child.raveWins += raveResult;
                }
            }
            // ----------------------------------------------------------------

            result = -result;           // flip sign going up the tree
            node   = node.parent;
        }
    }

    private IMove bestMove(Node root, IGameState fallback) {
        if (root.children.isEmpty()) {
            List<IMove> moves = fallback.getField().getAvailableMoves();
            return moves.get(ThreadLocalRandom.current().nextInt(moves.size()));
        }
        // Final selection: most-visited child (robust, ignores noise in win rates)
        return Collections.max(root.children, Comparator.comparingInt(n -> n.visits)).move;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private Node makeRoot(IGameState gameState) {
        Node root      = new Node();
        root.state     = gameState;
        root.mover     = 1 - botPlayer;   // opponent moved last to reach this state
        root.terminal  = false;
        root.untriedMoves = new ArrayList<>(gameState.getField().getAvailableMoves());
        return root;
    }

    private int evalMacro(String[][] macro, int player) {
        if (hasMacroWon(macro, player + ""))       return  1;
        if (hasMacroWon(macro, (1 - player) + "")) return -1;
        return 0;
    }

    private boolean hasMacroWon(String[][] macro, String mark) {
        for (int r = 0; r < 3; r++)
            if (macro[r][0].equals(mark) && macro[r][1].equals(mark) && macro[r][2].equals(mark))
                return true;
        for (int c = 0; c < 3; c++)
            if (macro[0][c].equals(mark) && macro[1][c].equals(mark) && macro[2][c].equals(mark))
                return true;
        if (macro[0][0].equals(mark) && macro[1][1].equals(mark) && macro[2][2].equals(mark)) return true;
        if (macro[0][2].equals(mark) && macro[1][1].equals(mark) && macro[2][0].equals(mark)) return true;
        return false;
    }

    private IMove heuristicMove(GameSimulator sim, List<IMove> moves) {
        String myMark  = sim.currentPlayer + "";
        String oppMark = (1 - sim.currentPlayer) + "";

        String[][] board = sim.getCurrentState().getField().getBoard();
        String[][] macro = sim.getCurrentState().getField().getMacroboard();

        IMove localWin  = null;
        IMove blockLocal = null;

        for (IMove m : moves) {
            if (winsLocal(board, m, myMark)) {
                int mx = m.getX() / 3, my = m.getY() / 3;
                if (wouldWinMacro(macro, mx, my, myMark)) return m;
                if (localWin == null) localWin = m;
            }
            if (blockLocal == null && winsLocal(board, m, oppMark)) {
                blockLocal = m;
            }
        }

        if (localWin  != null) return localWin;
        if (blockLocal != null) return blockLocal;

        for (IMove m : moves)
            if (m.getX() % 3 == 1 && m.getY() % 3 == 1) return m;

        for (IMove m : moves) {
            int lx = m.getX() % 3, ly = m.getY() % 3;
            if ((lx == 0 || lx == 2) && (ly == 0 || ly == 2)) return m;
        }

        return moves.get(ThreadLocalRandom.current().nextInt(moves.size()));
    }

    private boolean winsLocal(String[][] board, IMove m, String mark) {
        int x = m.getX(), y = m.getY();
        int sx = x - x % 3, sy = y - y % 3;
        int lx = x % 3,     ly = y % 3;

        String prev = board[x][y];
        board[x][y] = mark;
        boolean win = false;

        if (!win && board[sx][y].equals(mark) && board[sx+1][y].equals(mark) && board[sx+2][y].equals(mark))
            win = true;
        if (!win && board[x][sy].equals(mark) && board[x][sy+1].equals(mark) && board[x][sy+2].equals(mark))
            win = true;
        if (!win && lx == ly && board[sx][sy].equals(mark) && board[sx+1][sy+1].equals(mark) && board[sx+2][sy+2].equals(mark))
            win = true;
        if (!win && lx + ly == 2 && board[sx][sy+2].equals(mark) && board[sx+1][sy+1].equals(mark) && board[sx+2][sy].equals(mark))
            win = true;

        board[x][y] = prev;
        return win;
    }

    private boolean wouldWinMacro(String[][] macro, int mx, int my, String mark) {
        String prev     = macro[mx][my];
        macro[mx][my]   = mark;
        boolean win     = hasMacroWon(macro, mark);
        macro[mx][my]   = prev;
        return win;
    }

    // -----------------------------------------------------------------------
    // GameSimulator (inner class — unchanged from original)
    // -----------------------------------------------------------------------

    public enum GameOverState { Active, Win, Tie }

    private GameSimulator createSimulator(IGameState state) {
        GameSimulator sim = new GameSimulator(new GameState());
        sim.setCurrentPlayer(state.getMoveNumber() % 2);
        sim.getCurrentState().setRoundNumber(state.getRoundNumber());
        sim.getCurrentState().setMoveNumber(state.getMoveNumber());
        sim.getCurrentState().getField().setBoard(state.getField().getBoard());
        sim.getCurrentState().getField().setMacroboard(state.getField().getMacroboard());
        return sim;
    }

    class GameSimulator {
        private final IGameState currentState;
        int currentPlayer = 0;
        volatile GameOverState gameOver = GameOverState.Active;

        GameSimulator(IGameState s) { this.currentState = s; }

        void setCurrentPlayer(int p) { currentPlayer = p; }
        IGameState getCurrentState() { return currentState; }

        boolean updateGame(IMove move) {
            if (!verifyMoveLegality(move)) return false;
            updateBoard(move);
            currentPlayer = (currentPlayer + 1) % 2;
            return true;
        }

        private boolean verifyMoveLegality(IMove move) {
            IField f = currentState.getField();
            if (!f.isInActiveMicroboard(move.getX(), move.getY())) return false;
            if (move.getX() < 0 || move.getX() >= 9) return false;
            if (move.getY() < 0 || move.getY() >= 9) return false;
            if (!f.getBoard()[move.getX()][move.getY()].equals(IField.EMPTY_FIELD)) return false;
            return true;
        }

        private void updateBoard(IMove move) {
            String[][] board = currentState.getField().getBoard();
            board[move.getX()][move.getY()] = currentPlayer + "";
            currentState.setMoveNumber(currentState.getMoveNumber() + 1);
            if (currentState.getMoveNumber() % 2 == 0)
                currentState.setRoundNumber(currentState.getRoundNumber() + 1);
            checkAndUpdateIfWin(move);
            updateMacroboard(move);
        }

        private void checkAndUpdateIfWin(IMove move) {
            String[][] macro = currentState.getField().getMacroboard();
            int macroX = move.getX() / 3;
            int macroY = move.getY() / 3;

            if (!macro[macroX][macroY].equals(IField.EMPTY_FIELD) &&
                    !macro[macroX][macroY].equals(IField.AVAILABLE_FIELD)) return;

            String[][] board = currentState.getField().getBoard();
            String mark = currentPlayer + "";

            if (isWin(board, move, mark))
                macro[macroX][macroY] = mark;
            else if (isTie(board, move))
                macro[macroX][macroY] = "TIE";

            Move mm = new Move(macroX, macroY);
            if (isWin(macro, mm, mark)) gameOver = GameOverState.Win;
            else if (isTie(macro, mm)) gameOver = GameOverState.Tie;
        }

        private boolean isTie(String[][] board, IMove move) {
            int sx = move.getX() - move.getX() % 3;
            int sy = move.getY() - move.getY() % 3;
            for (int i = sx; i < sx + 3; i++)
                for (int j = sy; j < sy + 3; j++)
                    if (board[i][j].equals(IField.AVAILABLE_FIELD) ||
                            board[i][j].equals(IField.EMPTY_FIELD)) return false;
            return true;
        }

        boolean isWin(String[][] board, IMove move, String mark) {
            int lx = move.getX() % 3, ly = move.getY() % 3;
            int sx = move.getX() - lx, sy = move.getY() - ly;

            { int c = 0; for (int i = sy; i < sy+3; i++) if (board[move.getX()][i].equals(mark)) c++;
                if (c == 3) return true; }

            { int c = 0; for (int i = sx; i < sx+3; i++) if (board[i][move.getY()].equals(mark)) c++;
                if (c == 3) return true; }

            if (lx == ly) {
                int c = 0, yy = sy;
                for (int i = sx; i < sx+3; i++) if (board[i][yy++].equals(mark)) c++;
                if (c == 3) return true;
            }

            if (lx + ly == 2) {
                int c = 0, less = 0;
                for (int i = sx; i < sx+3; i++) if (board[i][(sy+2)-less++].equals(mark)) c++;
                if (c == 3) return true;
            }

            return false;
        }

        private void updateMacroboard(IMove move) {
            String[][] macro = currentState.getField().getMacroboard();

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    if (macro[i][j].equals(IField.AVAILABLE_FIELD))
                        macro[i][j] = IField.EMPTY_FIELD;

            int xt = move.getX() % 3;
            int yt = move.getY() % 3;

            if (macro[xt][yt].equals(IField.EMPTY_FIELD)) {
                macro[xt][yt] = IField.AVAILABLE_FIELD;
            } else {
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        if (macro[i][j].equals(IField.EMPTY_FIELD))
                            macro[i][j] = IField.AVAILABLE_FIELD;
            }
        }
    }
}