package dk.easv.bll.bot;

import dk.easv.bll.field.IField;
import dk.easv.bll.game.GameState;
import dk.easv.bll.game.IGameState;
import dk.easv.bll.move.IMove;
import dk.easv.bll.move.Move;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class TheBot implements IBot {

    private static final int TIME_BUFFER_MS = 50;
    private static final double UCT_C = 1.41;

    private int botPlayer;

    private static class Node {
        Node parent;
        List<Node> children = new ArrayList<>();
        List<IMove> untriedMoves = new ArrayList<>();
        IMove move;
        int mover;
        IGameState state;
        boolean terminal;
        int visits = 0;
        double wins = 0;
    }

    @Override
    public IMove doMove(IGameState gameState) {
        long endTime = System.currentTimeMillis() + gameState.getTimePerMove() - TIME_BUFFER_MS;
        botPlayer = gameState.getMoveNumber() % 2;

        Node root = makeRoot(gameState);

        while (System.currentTimeMillis() < endTime) {
            Node leaf = select(root);
            Node child = expand(leaf);
            int result = rollout(child);
            backpropagate(child, result);
        }

        return bestMove(root, gameState);
    }

    @Override
    public String getBotName() {
        return "TheBot";
    }

    private Node select(Node node) {
        while (!node.terminal && node.untriedMoves.isEmpty() && !node.children.isEmpty()) {
            node = bestUCT(node);
        }
        return node;
    }

    private Node bestUCT(Node parent) {
        double logParent = Math.log(parent.visits + 1);
        return Collections.max(parent.children, Comparator.comparingDouble(n ->
                (n.wins / (n.visits + 1e-9))
                        + UCT_C * Math.sqrt(logParent / (n.visits + 1e-9))
        ));
    }

    private Node expand(Node node) {
        if (node.terminal || node.untriedMoves.isEmpty()) {
            return node;
        }

        List<IMove> untried = node.untriedMoves;
        int idx = ThreadLocalRandom.current().nextInt(untried.size());
        IMove move = untried.remove(idx);

        GameSimulator sim = createSimulator(node.state);
        sim.updateGame(move);

        Node child = new Node();
        child.parent = node;
        child.move = move;
        child.mover = node.state.getMoveNumber() % 2;
        child.state = sim.getCurrentState();
        child.terminal = (sim.gameOver != GameOverState.Active);

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

    private int rollout(Node node) {
        if (node.terminal) {
            return evalMacro(node.state.getField().getMacroboard(), node.mover);
        }

        GameSimulator sim = createSimulator(node.state);
        int depth = 0;

        while (sim.gameOver == GameOverState.Active && depth < 200) {
            List<IMove> moves = sim.getCurrentState().getField().getAvailableMoves();
            if (moves.isEmpty()) break;
            sim.updateGame(heuristicMove(sim, moves));
            depth++;
        }

        if (sim.gameOver == GameOverState.Win) {
            int winner = (sim.currentPlayer + 1) % 2;
            return (winner == node.mover) ? 1 : -1;
        }
        return 0;
    }

    private void backpropagate(Node node, int result) {
        while (node != null) {
            node.visits++;
            node.wins += result;
            result = -result;
            node = node.parent;
        }
    }

    private IMove bestMove(Node root, IGameState fallback) {
        if (root.children.isEmpty()) {
            List<IMove> moves = fallback.getField().getAvailableMoves();
            return moves.get(ThreadLocalRandom.current().nextInt(moves.size()));
        }
        return Collections.max(root.children, Comparator.comparingInt(n -> n.visits)).move;
    }

    private Node makeRoot(IGameState gameState) {
        Node root = new Node();
        root.state = gameState;
        root.mover = 1 - botPlayer;
        root.terminal = false;
        root.untriedMoves = new ArrayList<>(gameState.getField().getAvailableMoves());
        return root;
    }

    private int evalMacro(String[][] macro, int player) {
        if (hasMacroWon(macro, player + "")) return 1;
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
        String myMark = sim.currentPlayer + "";
        String oppMark = (1 - sim.currentPlayer) + "";

        String[][] board = sim.getCurrentState().getField().getBoard();
        String[][] macro = sim.getCurrentState().getField().getMacroboard();

        IMove localWin = null;
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

        if (localWin != null) return localWin;
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
        int lx = x % 3, ly = y % 3;

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
        String prev = macro[mx][my];
        macro[mx][my] = mark;
        boolean win = hasMacroWon(macro, mark);
        macro[mx][my] = prev;
        return win;
    }

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