# Security Career — Next Steps (office se kar sakte ho, koi GPU/model kaam nahi)

Last updated: 2026-08-31

Context: Spica AI (bilingual LLM project, is repo ka main kaam) abhi ke liye
"done" maan liya hai — v4 SFT checkpoint final hai, backed up hai, EOS-stopping
bug bhi fix ho chuka hai. Ab focus system/network security career ki taraf
shift kar rahe hain, target: companies, government, Indian defence tak
pahunchna.

---

## 1. Accounts/registrations banao (aaj/is hafte)

- [ ] **TryHackMe** account banao — https://tryhackme.com
      "Pre Security" ya "Complete Beginner" learning path se shuru karo
- [ ] **HackTheBox** account banao (baad me use karoge, abhi register kar lo)
- [ ] **picoCTF** account banao — https://picoctf.org (beginner-friendly CTF)
- [ ] **CompTIA** account/profile check karo — Security+ ya Network+ exam
      registration ke liye (pehle decide karo kaunsa pehle: agar networking
      basics weak hain, Network+ pehle; warna seedha Security+)

## 2. Study material dhundo (registration/purchase, actual padhna baad me)

- [ ] Security+ / Network+ study guide (book ya online course — Udemy,
      Professor Messer (free YouTube series bhi hai Security+ ke liye) dekho
- [ ] OSCP ka syllabus/prerequisites padh lo (abhi lena nahi, bas plan
      samajhna) — https://www.offsec.com/courses/pen-200/

## 3. Home lab setup (office pe download/planning ho sakta hai, setup ghar pe)

- [ ] VirtualBox ya VMware Workstation download karo
- [ ] Kali Linux ISO download karo (attacker machine ke liye)
- [ ] Metasploitable VM download karo (practice target ke liye)

## 4. Community/reference (bookmark kar lo)

- [ ] r/netsec (Reddit)
- [ ] Nullcon (Indian security conference) — website check karo, upcoming
      events dekho — https://nullcon.net
- [ ] c0c0n (Kerala Police + community security conference)

## 5. Government/Defence path (research/reading, action baad me)

- [ ] CERT-In empanelled security auditing firms ki list dekho (Google:
      "CERT-In empanelled organisations list") — ye ek accessible entry
      point hai direct sarkari naukri se
- [ ] DRDO recruitment page bookmark karo, eligibility criteria padho
- [ ] GeM (Government e-Marketplace) pe security-related tenders/companies
      dekho — samajhne ke liye kaunsi private companies govt contracts
      lekar security kaam karti hain

---

## Timeline reference (jo pehle discuss kiya tha)

- **Is hafte**: TryHackMe account + shuru karo, Network+/Security+ decide karo
- **2-4 hafte**: home lab setup, picoCTF try karo
- **2-3 mahine**: Security+ exam de do, OSCP prep plan banao
- **Ongoing**: community follow karo, conferences ke baare me seekho

---

## Spica AI status (reference, agar kabhi wapas aana ho)

- Repo: https://github.com/hostspicaindia/spica-ai
- Best checkpoint: `latest_500m_sft_v4_final.pt` (local machine pe, backed up)
- Pretrain checkpoint: `latest_500m_pretrain_v2_final.pt` (local machine pe)
- vast.ai instance: delete kar diya (cost bachane ke liye) — sab kuch
  reproducible hai code + backed-up checkpoints se
- Last real fix: EOS-stopping bug (commit `aa60e1c`) — generate() ab
  correctly rukta hai SFT checkpoint ke response ke baad, "sahi answer ke
  baad garbage" wala pattern fix ho gaya
