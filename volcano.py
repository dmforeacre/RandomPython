# Author: Daniel Foreacre
# Date: Nov 5, 2021
# Class: CSCI 450 Artificial Intelligence Dr Schwartz
# Assignment: Konane Minimax with Alpha/Beta Pruning
# Description: Implements a minimax algorithm with alpha/beta
#              pruning to control a player's moves in the
#              game of konane.
#              This code I *did* write, but extends the classes Konane and Player
#              which were *not* written by me.

from konane import *

from konane import Konane
from konane import Player
from konane import RandomPlayer
from konane import SimplePlayer

class Move():
   def __init__(self, move, value):
      self.move = move
      self.value = value

class MinimaxPlayer(Konane, Player):
 def __init__(self, size, depthLimit):
    Konane.__init__(self, size)
    self.limit = depthLimit
 def initialize(self, side):
    self.name = "Bahamut"
    self.side = side
 def getMove(self, board):
    moves = self.generateMoves(board, self.side)
    n = len(moves)
    if n == 0:
      return []
    val = -10000
    a = -10000
    b = 10000
    bestMove = moves[0]
    for m in moves:
         nextVal = self.minimax(self.nextBoard(board, self.side, m), self.limit - 1, self.opponent(self.side), a, b)
         if nextVal > val:
            val = nextVal
            bestMove = m
         if val > a:
            a = val
         if val >= b:
            break
    return bestMove
      
        
 def minimax(self, board, depth, side, a, b):
   # if depth limit reached, evaluate the board
   if depth == 0:
      return self.eval(board)
   moves = self.generateMoves(board, side)
   # if no moves found, player lost
   if len(moves) == 0:
      if side == self.side:
         val = -10000#-float("inf")
      else:
         val = 10000#float("inf")

   # Check a/b values for other nodes for potential pruning
   if side == self.side:
      val = -10000
      for m in moves:
         nextVal = self.minimax(self.nextBoard(board, side, m), depth - 1, self.opponent(side), a, b)
         if nextVal > val:
            val = nextVal
         if val > a:
            a = val
         if val >= b:
            break
      return val
   else:
      val = 10000
      for m in moves:
         nextVal = self.minimax(self.nextBoard(board, side, m), depth - 1, self.opponent(side), a, b)
         if nextVal < val:
            val = nextVal
         if val < b:
            b = val
         if val <= a:
            break
      return val
            
 def eval(self, board):
    """
    Because the goal of the game is to force the opponent to run out of moves, this evaluation
    function is based on the number of moves the opponent has for the given board. Since the
    player also wants to have as many potential moves as possible this is also factored into
    the evaluation. So the value returned is the ratio of the player's moves to the opponent's
    moves, multiplied by 100 to provide better integer resolution.
    Values over 100 are beneficial for Max
    Values under 100 are beneficial for Min
    """
    moves = len(self.generateMoves(board, self.side))
    oppMoves = len(self.generateMoves(board, self.opponent(self.side)))
    if oppMoves != 0:
      return int((moves / oppMoves) * 100)
    else:
      return 10000

game = Konane(6)
game.playNGames(1, MinimaxPlayer(6,4), HumanPlayer(), True)
