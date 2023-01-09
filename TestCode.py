
import time
# import pandas as pd
# Happy '기쁨' = 0                    
# Fear '불안' = 1                   
# Embarrassed '당황' = 2                    
# Sad '슬픔' = 3                    
# Rage '분노' = 4                    
# Hurt '상처' = 5  

start = time.time()

# rows = pd.read_csv('movie_review_long.csv')



if __name__ == "__main__":
    # [array([ 5.8531494, -0.6694658, -1.1817917, -0.908142 , -1.2727499, -1.5652826], dtype=float32), 'Happy', 5.8531494]
    print(analyze_word("아무런기대도하지않았지만생각보다괜찮은결과였다"))
    print(analyze_word("이게맞아??이런방식은좀아닌거같아"))
    print(analyze_word("으...의도는좋은거같지만좀망한거같은디??"))
    print(analyze_word("할 일이너무많네요😅할 일은 항상 끝이 없….🫠"))
    print(analyze_word("비둘기, 라이너 릴케 소학교 불러 다하지 봄이 슬퍼하는 봅니다.한이제벌레는북간도에까닭입니다.시와하나에차이름을나는묻힌딴은봅니다.이름과,불러우는 다하지 어머니, 북간도에 거외다. 별 추억과 멀듯이, 토끼, 아름다운 있습니다. 이름과 많은 헤는 어머님, 때 이런 피어나듯이 아침이 속의 듯합니다. 별 같이 강아지, 별을 별빛이 걱정도 별 당신은 있습니다. 무성할 어머님, 같이 밤을 프랑시스 피어나듯이 비둘기, 이름을 봅니다. 까닭이요, 소녀들의 불러 동경과 이웃 있습니다. 이름과 나의 별 나의 이름자 있습니다. 쉬이 못 하나에 마디씩 별에도 아직 내일 버리었습니다."))
    # for row in rows.itertuples():
    #     print(analyze_word(row.text))


    print("소요시간 :", time.time() - start) 