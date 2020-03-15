#
# CS1010FC --- Programming Methodology
#
# Mission N Solutions
#
# Note that written answers are commented out to allow us to run your
# code easily while grading your problem set.
from random import *

#######
#Task 1a#
#######

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 1 mark for creating the correct matrix

def new_game(n):
    matrix = []

    for i in range(n):
        matrix.append([0] * n)
    return matrix

###########
# Task 1b #
###########

# [Marking Scheme]
# Points to note:
# Must ensure that it is created on a zero entry
# 1 mark for creating the correct loop
#随机在空格处生成一个2
def add_two(mat):
    a=randint(0,len(mat)-1)#上下限
    b=randint(0,len(mat)-1)
    while(mat[a][b]!=0):
        a=randint(0,len(mat)-1)
        b=randint(0,len(mat)-1)
    mat[a][b]=2
    return mat

###########
# Task 1c #
###########

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 0 marks for completely wrong solutions
# 1 mark for getting only one condition correct
# 2 marks for getting two of the three conditions
# 3 marks for correct checking

def game_state(mat):
    ##得到2048则赢了
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j]==256:
                return 'win'
    ##存在可合并的，游戏继续，检查其右相邻和下相邻
    for i in range(len(mat)-1): #intentionally reduced to check the row on the right and below
        for j in range(len(mat[0])-1): #more elegant to use exceptions but most likely this will be their solution
            if mat[i][j]==mat[i+1][j] or mat[i][j+1]==mat[i][j]:
                return 'not over'
    ##存在空格，游戏继续
    for i in range(len(mat)): #check for any zero entries
        for j in range(len(mat[0])):
            if mat[i][j]==0:
                return 'not over'
    #最后一行是否有可合并
    for k in range(len(mat)-1): #to check the left/right entries on the last row
        if mat[len(mat)-1][k]==mat[len(mat)-1][k+1]:
            return 'not over'
    #最后一列是否有可合并的
    for j in range(len(mat)-1): #check up/down entries on last column
        if mat[j][len(mat)-1]==mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'

###########
# Task 2a #
###########

# [Marking Scheme]
# Points to note:
# 0 marks for completely incorrect solutions
# 1 mark for solutions that show general understanding
# 2 marks for correct solutions that work for all sizes of matrices
#左右互换
def reverse(mat):
    new=[]
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0])-j-1])
    return new

###########
# Task 2b #
###########

# [Marking Scheme]
# Points to note:
# 0 marks for completely incorrect solutions
# 1 mark for solutions that show general understanding
# 2 marks for correct solutions that work for all sizes of matrices
#行列互换
def transpose(mat):
    new=[]
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new

##########
# Task 3 #
##########

# [Marking Scheme]
# Points to note:
# The way to do movement is compress -> merge -> compress again
# Basically if they can solve one side, and use transpose and reverse correctly they should
# be able to solve the entire thing just by flipping the matrix around
# No idea how to grade this one at the moment. I have it pegged to 8 (which gives you like,
# 2 per up/down/left/right?) But if you get one correct likely to get all correct so...
# Check the down one. Reverse/transpose if ordered wrongly will give you wrong result.
#把0移动右边
def cover_up(mat):
    new=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    done=False
    for i in range(4):
        count=0
        for j in range(4):
            if mat[i][j]!=0:
                new[i][count]=mat[i][j]
                if j!=count:
                    done=True
                count+=1
    return (new,done)
#向左合并
def merge(mat):
    done=False
    sum=0
    for i in range(4):
         for j in range(3):
             if mat[i][j]==mat[i][j+1] and mat[i][j]!=0:
                 mat[i][j]*=2
                 sum+=mat[i][j]
                 mat[i][j+1]=0
                 done=True
    return (mat,done,sum)


def up(game):
        #print("logic up")
        # return matrix after shifting up
        game=transpose(game)
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        raward = temp[2]
        game=cover_up(game)[0]
        game=transpose(game)
        #print("logic up game,done,raward", game, done, raward)
        return (game,done,raward)

def down(game):
        #print("logic down")
        game=reverse(transpose(game))
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        raward = temp[2]
        game=cover_up(game)[0]
        game=transpose(reverse(game))
        #print("logic down game,done,raward", game, done, raward)
        return (game,done,raward)

def left(game):
        #print("logic left")
        # return matrix after shifting left
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        raward = temp[2]
        game=cover_up(game)[0]
        #print("logic left game,done,raward", game, done, raward)
        return (game,done,raward)

def right(game):
        #print("logic right")
        # return matrix after shifting right
        game=reverse(game)
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        raward = temp[2]
        game=cover_up(game)[0]
        game=reverse(game)
        #print("logic right game,done,raward",game,done,raward)
        return (game,done,raward)

def max_mat(matrix):
    max_sum = matrix[0][0]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if(max_sum<matrix[i][j]):
                max_sum = matrix[i][j]
    return max_sum
def max_mat_x_y(matrix):
    max_sum = matrix[0][0]
    x=1
    y=1
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if(max_sum<matrix[i][j]):
                max_sum = matrix[i][j]
                x=i+1
                y=j+1
    return (max_sum,x,y)
#每一行能合并数的和
def merge_sum(mat):
    sum_row=[0,0,0,0]
    for i in range(4):
        for j in range(3):
            if(mat[i][j]==mat[i][j+1]):
                sum_row[i]+=(2*mat[i][j])
                mat[i][j+1]=0
    return sum_row

def new_s(game):
    s = []
    temp_game1=cover_up(game)[0]
    sum_row=merge_sum(temp_game1)
    for i in range(len(sum_row)):
        s.append(sum_row[i])
    temp_game2=transpose(game)
    temp_game2 = cover_up(temp_game2)[0]
    sum_colu=merge_sum(temp_game2)
    for i in range(len(sum_colu)):
        s.append(sum_colu[i])
    max_sum, x, y=max_mat_x_y(game)
    s.append(max_sum)
    s.append(x)
    s.append(y)
    return s

def unification(s):
    max=0
    for i in range(len(s)):
        if max<s[i]:
            max=s[i]
    for i in range(len(s)):
        s[i]=s[i]/max
    return s


def flatten_list(matrix):
    s=[]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            s.append(matrix[i][j])
    return s