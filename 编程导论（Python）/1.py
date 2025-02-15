worldlist=["shit","fuck","dick","damn","hell","god","freek","pondcum"]
import random
real_word=random.choice(worldlist)
hidden_word="*"*len(real_word)
count=0
real_count=0
print(f"the guess word is:{hidden_word}")
while count<len(real_word)*2:
    guess_letter=input("please input a letter:")
    if guess_letter in real_word:
        count+=1
        real_count+=1
        for k in range(0,len(real_word)):
            if real_word[k]==guess_letter:
                a=k+1
                hidden_word=hidden_word[:k]+guess_letter+hidden_word[a:]
        if "*"in hidden_word:
            print(hidden_word,",guess right,try again")
            print("_"*30)
        else:
                print(hidden_word,",guess right,you win!")
                break
    else:
        count+=1
        if count<len(real_word)*2:
            print(hidden_word,",you guess wrong,try again!")
            print("_"*30)
        else:  
            print("you lost the game,game over!")
    
