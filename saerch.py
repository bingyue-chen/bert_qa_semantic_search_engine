# -*- coding: utf-8 -*-

import numpy as np
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer


if __name__ == '__main__':
	"""
	setting questions
	"""
	questions = ["What is Giftpack?How to use it?","How do I sign up?","Are you open on holidays and weekends? Will shipping costs be different?","What are the available cities?","What cities are going to be available in the future?","I want Giftpack in more cities. How can I let you know?","Where can I find my previous gifting records?","Will there be a discounted price if I want to deliver a huge amount of self-preparation presents at once?","How should I contact your reception if I want to customize a wedding present?","How do I pay for the gift?","What is The Magic tip Box?","Can I send a gift that needs to be pre-ordered earlier?","If I choose several gifts, can I combine shipping costs?","Can gifts be delivered overseas?","Can you help me buy gifts?","Can I track my delivery process?","If there is a mistake in my order, what can I do to correct it?","After placing the order, can I change the gift I want to send?","Can I change the designated gifting time after placing the order?","Can I get a refund if I don't need the gift after placing an order?","What if the gift doesn't arrive at the assigned time?","What will happen, if the gift is rejected by the receiver?","How can I contact my Packer?","Can I or the receiver leave a comment or review a gift and service?","If the gift receiver does not answer the phone, what will Giftpack do?","If the receiver is in another location, will the Packer send the gift there for me?","Can Packer wait for the receiver there for half an hour?","Do I get a discount for my first order?","What is Giftpack's Newsletter? (What can I get from subscribing to the Newsletter?)","What is Giftpack Credit? How to get it?","How to get Coupon Code?","How to use my Coupon Code?","Why do I have to fill in all of my personal information?","How can I update my profile?","I’ve forgotten my password!!","Why haven’t I received my verification code?","I am having trouble launching the Giftpack website. How can I contact customer support?","My Giftpack App keeps crashing. Is there a way to fix this problem?","How can I switch languages on the Giftpack website?","How can I support Giftpack?","Giftpack究竟是什麼服務？我可以如何使用？","如何加入Giftpack？","Giftpack的營業時間為何？","目前Giftpack開放哪些地區呢？","未來有哪些地區會開放？","我想建議Giftpack開放更多城市，該怎麼做呢？","什麼是送禮紀錄？哪裡看得到？","Giftpack有提供 個人多筆／企業送禮／VIP 客製訂單嗎？","在Giftpack上無法找到符合我需求的禮品或送禮服務，我可以向誰聯繫呢？","目前Giftpack支援哪些付費方式？","什麼是「神奇小費箱」？","有些禮物需要提前預訂，但我想提前送達是可以的嗎？","如果我要選多種禮物，運費可以合一嗎？","禮物可以做到跨國郵寄服務嗎？","可以自己準備好禮物或是前預訂，請你們幫我代送嗎？","我該如何得知送禮進度？","我發現訂單內容有錯誤，該如何做修正？","我可以更改要送的禮物嗎？","我可以更改送禮時間嗎？","下單之後，發現不需要了，可以退款嗎？","如果禮物若未能準時抵達收禮者的手中怎麼辦？","如果禮物送達後，被收禮者拒收該怎辦？","該如何聯絡外送員呢？","使用者可以為產品進行評論嗎？","若送禮時聯繫不到收禮人，我可以有哪些處理方式呢？","承上題，發現收禮人不在我填寫的收禮地址，而是在另一個地點，快遞員會幫我送過去嗎？","承上題，若收禮人於預定時間並不在送禮地址，經聯繫後需要等候半小時以上，快遞員會幫我等嗎？需要額外收費嗎？","何謂首購優惠？","什麼是Giftpack電子報？","聽說註冊Giftpack還可以賺得購物金，是真的嗎？","我要如何得知最新的優惠資訊呢？","優惠碼該如何使用？","為什麼必須填寫完整個人資料？","哪裡可以更新我的個人檔案？","忘記密碼怎麼辦？","為什麼收不到簡訊驗證碼？","網站為什麼無法瀏覽？該如何尋求即時協助？","APP不停閃退，我該如何及時解決？","我該如何轉換現在瀏覽的語言？","我該如何贊助Giftpack"]

	"""
	get questions vector
	"""
	bc = BertClient()
	questions_vecs = bc.encode(questions)

	"""
	do query
	"""
	continue_search = True
	while continue_search:
	    query = input('your question: ')
	    
	    if query == "exit":
	    	continue_search = False
	    	continue
	    
	    query_vec = bc.encode([query])[0]
	    # compute normalized dot product as score
	    score = np.sum(query_vec * questions_vecs, axis=1) / np.linalg.norm(questions_vecs, axis=1)
	    topk_idx = np.argsort(score)[::-1][:5]
	    for idx in topk_idx:
	        print('> %s\t%s' % (score[idx], questions[idx]))

	    print('')