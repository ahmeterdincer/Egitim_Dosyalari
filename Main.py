#Age - çalışanların yaşı 
#department - calışanların departmanları
#Distance_From_Home - çalışanların evlerinin iş yerlerine uzaklıkları
#Years_At_Company - çalışanların şirkette kaç yıl çalıştıgı
#Average_Monthly_Hours - çalışanların aylık ortalama çalıma saatleri
#Number_Of_Projects - çalışanın dahil oldugu proje sayısı
#Satisfaction_Level - çalışanın işinden memnuniyet puanı(0-1)
#Last_Evaluation - yönetiiciler tarafından yapılan calısanların degerlendirme sonucları(0-1)
#Salary - çalışanların maaşı
#Attrition - çalışanın şirketten ayrılıp ayrılmadıgının sonucu 0 veya 1

import numpy as np 
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Veriyi oku
print("-"*60)
print("Veri yükleme...")
print("-"*60)

dosya_yolu = r"C:\Users\EXCALIBUR\OneDrive\Masaüstü\kulüpVol2\HR_Analytics.csv"
veri=pd.read_csv(dosya_yolu)
print(type(veri))
print("\nİlk 5 satır...\n---------------------")
print(veri.head())

#Veri ön işleme
print("-"*60)
print("Veri ön işleme...")
print("-"*60)

#Eksik veri kontrolu
print("Eksik veri kontrolu\n ---------------------------")
print(veri.isnull().sum())
#Eger eksik veri olsaydı ortalama veya medyan ile doldurabilirdik

#Kategorik verileri sayısallaştırma(encode etme)
from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
veri["Departman_Kod"] = encoder.fit_transform(veri["Department"])
print("\nDepartman - Kod eslemesi:", dict(zip( #dict sozluk olsuturuyor. zip iki listeyi yanyana birleştirme yapıyor
    encoder.classes_, #.classes_ verilerde kullanılan engineer gibi isimleri tutuyor
    encoder.transform(encoder.classes_)
)))


#Ozellikler (X) ve hedefler (y) ayrimi
#Regresyon hedefi     : Salary   (maas tahmini)
#Siniflandirma hedefi : Attrition (0=Kaldi, 1=Ayrildi)

ozellikler = ["Age", "Departman_Kod", "Distance_From_Home",
              "Years_At_Company", "Average_Monthly_Hours",
              "Number_Of_Projects", "Satisfaction_Level", "Last_Evaluation"]

X = veri[ozellikler]
y_regresyon = veri["Salary"]
y_siniflandirma=veri["Attrition"]

#Ölçeklendirme (KNN ve SVM için)
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_olcekli = scaler.fit_transform(X)
print("\nÖn işleme tamamlandı\n------------------------")

#Regresyon
print("-"*60)
print("REGRESYON...")
print("-"*60)

#Regresyon için egitim ve test verilerini ayrıştırılması
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y_regresyon,test_size=0.2,random_state=42)

#Basit Doğrusal Regresyon
#Sadece Age özelliğini kullanıcaz
print("\n Basit Doğrusal Regresyon \n-------------------------")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#r2_score — R² (Açıklama Oranı)
#Modelin veriyi ne kadar iyi açıkladığını 0 ile 1 arasında bir skorla gösterir.
#R² = 0.0  → Model hiçbir şeyi açıklayamıyor, berbat
#R² = 0.5  → Verinin yarısını açıklıyor, orta
#R² = 1.0  → Mükemmel, tüm veriyi açıklıyor

#mean_squared_error — MSE (Ortalama Kare Hata)
#Tahmin ile gerçek değer arasındaki ortalama hatayı gösterir. Ne kadar küçükse o kadar iyi.
#Gerçek maaş   : 12.000
#Tahmin edilen : 11.500
#Fark          : 500  → karesini alır → 250.000


# Tek sutun sec (2D hale getirmek icin kose parantez ile)
X_age_train = X_train[["Age"]]
X_age_test   = X_test[["Age"]]
# Modeli egitme
model_slr = LinearRegression()
model_slr.fit(X_age_train, y_train)       
tahmin_slr=model_slr.predict(X_age_test)

#Değerlendirme
mse_slr=mean_squared_error(y_test,tahmin_slr)
r2_slr=r2_score(y_test,tahmin_slr)
print(f"  R2 Skoru : {r2_slr:.4f}")
print(f"  MSE      : {mse_slr:.2f}")
print(f"  Katsayi  : {model_slr.coef_[0]:.2f}  |  Sabit: {model_slr.intercept_:.2f} --->> y={model_slr.coef_[0]:.2f}.age + {model_slr.intercept_:.2f}")

#Görselleştirme
# Grafik: Gercek noktalar (mavi) + Tahmin cizgisi (kirmizi)
plt.figure(figsize=(7, 4))
plt.scatter(X_age_test, y_test, color="steelblue", alpha=0.4, label="Gercek")
plt.plot(sorted(X_age_test["Age"]),
         model_slr.predict(X_age_test.sort_values("Age")),
         color="red", linewidth=2, label="Tahmin")
plt.title("Simple Linear Regression: Yas - Maas")
plt.xlabel("Yas")
plt.ylabel("Maas")
plt.legend()
plt.tight_layout()
plt.savefig("simple_linear_regresyon.png")
plt.close()
print("  Grafik kaydedildi: simple_linear_regresyon.png")

#Çoklu linear regresyon
#model oluşturma
print("\n COKLU LINEAR REGRESSION\n--------------------------")
model_mlr=LinearRegression()
model_mlr.fit(X_train,y_train)
tahmin_mlr=model_mlr.predict(X_test)
#model değerlendirme
mse_mlr= mean_squared_error(y_test,tahmin_mlr)
r2_mlr=r2_score(y_test,tahmin_mlr)
print(f"R2 skoru : {r2_mlr:.4f}")
print(f"mse skoru : {mse_mlr:.2f}")

#Polinom regresyon
#model oluşturma
print("\n Polinom Regresyon \n--------------------------------")
from sklearn.preprocessing import PolynomialFeatures
polinomlastirma = PolynomialFeatures(degree=2)
X_poli_train= polinomlastirma.fit_transform(X_train)
X_poli_test=polinomlastirma.transform(X_test)
#Burada değişkenlerle biraz oynayarak linearregression modelini kandırıyoruz.
#linearregression modeli sadece düz bir çizgi çekmeye programlıdır
#bu yüzden verilerle oynuyoruz aslında elimizdeki verinin karesini almamız gerekiyor ama biz o kare
# sonucunu alıp tek dereceli bir değişkene atıyoruz ve model tek dereceli bir değişken kullandıgını sanıyo
# peki model sadece düz çizgi çekmeye programlıysa nasıl oluyorda eğrili bir grafik elde ediyoruz 
#aslında model gercekten düz bir çizgi çekiyor ama bunun bize yansıması dalgalı bir grafik oluyo 
#bunu bir kagida duz bir cizgi cektikten sonra kagıdı burusturdugunuzda o duz cizginin dalgalı gorunmesine benzetebiliriz

model_plr=LinearRegression()
model_plr.fit(X_poli_train,y_train)
tahmin_plr=model_plr.predict(X_poli_test)

mse_poli = mean_squared_error(y_test, tahmin_plr)
r2_poli  = r2_score(y_test, tahmin_plr)

print(f"  R2 Skoru : {r2_poli:.4f}")
print(f"  MSE      : {mse_poli:.2f}")

# Regresyon karsilastirma grafigi
etiketler_reg = ["Simple LR", "Multiple LR", "Polynomial"]
r2_degerleri  = [r2_slr, r2_mlr, r2_poli]

plt.figure(figsize=(7, 4))
plt.bar(etiketler_reg, r2_degerleri, color=["blue", "green", "red"])
plt.ylim(0, 1)
plt.title("Regresyon Modelleri - R2 Karsilastirmasi")
plt.ylabel("R2 Skoru")
for i, v in enumerate(r2_degerleri):
    plt.text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("regresyon_karsilastirma.png")
plt.close()
print("\n  Grafik kaydedildi: regresyon_karsilastirma.png")

#-----------------------------------
#Sınıflandırma

print("\n" + "=" * 60)
print("SINIFLANDIRMA")
print("=" * 60)

#Sınıflandırma için egitim ve test verilerinin bölünmesi(scale edilmis X kullanıyoruz)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_egitim_sin, X_test_sin, y_egitim_sin, y_test_sin = train_test_split( X_olcekli, y_siniflandirma, test_size=0.2, random_state=42)


#lojistik regresyon

print("\n--- B-1: Logistic Regression ---")
from sklearn.linear_model import LogisticRegression

model_lojistik = LogisticRegression(max_iter=1000, random_state=42)
model_lojistik.fit(X_egitim_sin, y_egitim_sin)
tahmin_lojistik = model_lojistik.predict(X_test_sin)
print(f"  Dogruluk         : {accuracy_score(y_test_sin, tahmin_lojistik):.4f}")
print(f"  Karmasiklik Matrisi:\n{confusion_matrix(y_test_sin, tahmin_lojistik)}")


#KNN K en yakın komşu

print("\nKNN\n----------------------------")
from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_egitim_sin, y_egitim_sin)
tahmin_knn = model_knn.predict(X_test_sin)
print(f"  Dogruluk         : {accuracy_score(y_test_sin, tahmin_knn):.4f}")
print(f"  Karmasiklik Matrisi:\n{confusion_matrix(y_test_sin, tahmin_knn)}")



# SVM SUPPORT VECTOR MACHINE

print("\nSVM\n ------------------------")
from sklearn.svm import SVC

model_svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
#kernel verinin karar sınırının şeklini belirtiyor rbf kıvrımlı bir yapı sınır oluşturuyor.
# örnegin lineerde var ama  dogrusal veriler için uygun

#C Ceza katsayısıdır model öğrenirken hata yaptıgında dönecek tepkiyi belirliyor
#eğer çok düşükse undefitting(eksik öğrenme) riskini artırır. Yüksek ise aşırı öğrenme riskini artırır

#gamma = verilerin birbirilerine etki ağırlıklarını ayarylıyor 
#gamma kucukse veri agırşıgı dusuktur ve her bir veri o veriyi etkilemeye baslar
#buyukse de sadece etrafında bir iki tane veriyle etkileşime giriyor
model_svm.fit(X_egitim_sin, y_egitim_sin)
tahmin_svm = model_svm.predict(X_test_sin)
print(f"  Dogruluk         : {accuracy_score(y_test_sin, tahmin_svm):.4f}")
print(f"  Karmasiklik Matrisi:\n{confusion_matrix(y_test_sin, tahmin_svm)}")


#B-4  KARAR AGACI (Decision Tree)

print("\nDecision Tree \n--------------")
from sklearn.tree import DecisionTreeClassifier
#max depthi deneye deneye bulalım elimizdeki ozellik ve veri sayısı az zaten
#bunu bulmaya yarayan algoritmalarda deneyerek buluyor en son sonucları karsılastırıp hangisi daha iyi onu alıyor
model_karar_agaci = DecisionTreeClassifier(max_depth=5, random_state=42)
model_karar_agaci.fit(X_egitim_sin, y_egitim_sin)
tahmin_karar_agaci = model_karar_agaci.predict(X_test_sin)
print(f"  Dogruluk         : {accuracy_score(y_test_sin, tahmin_karar_agaci):.4f}")
print(f"  Karmasiklik Matrisi:\n{confusion_matrix(y_test_sin, tahmin_karar_agaci)}")


#RANDOM FOREST

print("\nRandom Forest\n ---------------")
from sklearn.ensemble import RandomForestClassifier
#n_estmators parametresini tıpkı deciison treedeki depth gibi deneeyerek buluyoruz
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_egitim_sin, y_egitim_sin)
tahmin_rf = model_rf.predict(X_test_sin)
print(f"  Dogruluk         : {accuracy_score(y_test_sin, tahmin_rf):.4f}")
print(f"  Karmasiklik Matrisi:\n{confusion_matrix(y_test_sin, tahmin_rf)}")


# Siniflandirma karsilastirma grafigi
etiketler_sin = ["Logistic\nReg.", "KNN", "SVM", "Decision\nTree", "Random\nForest"]
dogruluklar   = [
    accuracy_score(y_test_sin, tahmin_lojistik),
    accuracy_score(y_test_sin, tahmin_knn),
    accuracy_score(y_test_sin, tahmin_svm),
    accuracy_score(y_test_sin, tahmin_karar_agaci),
    accuracy_score(y_test_sin, tahmin_rf),
]

plt.figure(figsize=(9, 4))
renkler = ["blue", "green", "red", "purple", "orange"]
plt.bar(etiketler_sin, dogruluklar, color=renkler)
plt.ylim(0, 1)
plt.title("Siniflandirma Modelleri - Dogruluk Karsilastirmasi")
plt.ylabel("Dogruluk (Accuracy)")
for i, v in enumerate(dogruluklar):
    plt.text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold", fontsize=9)
plt.tight_layout()
plt.savefig("siniflandirma_karsilastirma.png")
plt.close()
print("\n  Grafik kaydedildi: siniflandirma_karsilastirma.png")



#KÜMELEME -- K-Means

print("\n" + "=" * 60)
print("K-Means Algoritmasi...")
print("=" * 60)

# # Kumeleme için bu iki özellik alınıyor ki görselleştirilebilsin 
X_kume = veri[["Satisfaction_Level", "Average_Monthly_Hours"]].values

# # Elbow (dirsek) yontemi ile en iyi k sayisini bulalim
toplam_hata = []
k_aralik = range(1, 11)
# 1-11 seçilmesinin sebebi biraz da keyfi bir seçim cıktı olarak istedigimiz durumlar asagı yukarı belli 
# memnun mu degil mi kararsız mı gibi kümelenebilir o yuzden 1 ile keyfi bir aralık verdik
from sklearn.cluster import KMeans
for k in k_aralik:
    kmeans_deneme = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_deneme.fit(X_kume)
    toplam_hata.append(kmeans_deneme.inertia_) #inertia verilerin ne kadar sıkı veya derli toplu olduklarınıa göre değer dondurur

plt.figure(figsize=(7, 4))
plt.plot(list(k_aralik), toplam_hata, marker="o", color="steelblue")
plt.title("Elbow Yontemi - En Iyi Kume Sayisi")
plt.xlabel("Kume Sayisi (k)")
plt.ylabel("Inertia (toplam hata)")
plt.xticks(list(k_aralik))
plt.tight_layout()
plt.savefig("elbow_grafigi.png")
plt.close()
print("  Grafik kaydedildi: elbow_grafigi.png")

# # K=3 ile nihai modeli egit
en_iyi_k = 3
model_kmeans = KMeans(n_clusters=en_iyi_k, random_state=42, n_init=10)
model_kmeans.fit(X_kume)

kume_etiketleri = model_kmeans.labels_
merkezler       = model_kmeans.cluster_centers_



print(f"\n  Secilen k : {en_iyi_k}")
print("  Kume boyutlari:")
for k in range(en_iyi_k):
    print(f"    Kume {k}: {np.sum(kume_etiketleri == k)} calisan")

# # Kume gorsellestime
renkler_kume = ["red", "green", "blue"]
plt.figure(figsize=(8, 5))
for k in range(en_iyi_k):
    noktalar = X_kume[kume_etiketleri == k]
    plt.scatter(noktalar[:, 0], noktalar[:, 1],
                c=renkler_kume[k], label=f"Kume {k}", alpha=0.5, s=30)

plt.scatter(merkezler[:, 0], merkezler[:, 1],
            c="black", marker="X", s=200, label="Merkez", zorder=5)
plt.title("K-Means Kumeleme: Memnuniyet - Calisma Saati")
plt.xlabel("Memnuniyet Skoru")
plt.ylabel("Aylik Calisma Saati")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_kumeleme.png")
plt.close()
print("  Grafik kaydedildi: kmeans_kumeleme.png")



# OZET - TUM MODEL SONUCLARI

print("\n" + "=" * 60)
print("  OZET - TUM MODEL SONUCLARI")
print("=" * 60)

print("\n  [ REGRESYON ]")
print(f"  Simple Linear   -> R2: {r2_slr:.4f}")
print(f"  Multiple Linear -> R2: {r2_mlr:.4f}")
print(f"  Polynomial      -> R2: {r2_poli:.4f}")

print("\n  [ SINIFLANDIRMA ]")
isim_listesi = ["Logistic Reg.", "KNN", "SVM", "Decision Tree", "Random Forest"]
for isim, dogr in zip(isim_listesi, dogruluklar):
    print(f"  {isim:<20} -> Dogruluk: {dogr:.4f}")

from sklearn.metrics import silhouette_score
sil_skoru = silhouette_score(X_kume, kume_etiketleri)
#silhoutte score verilerin aynı küme içerisindekilere ne kadar benziyor farklı kümelerdekiyle ne kadar ayrısıyor diye bakar
#ve sonuç 1-ÇOK İYİ 0- ORTALAMA -1-KÖTÜ 
print("\n  [ KUMELEME ]")
print(f"  K-Means (k={en_iyi_k}) basariyla tamamlandi. (Silhouette Skoru: {sil_skoru:.4f})")
print("\n  Tum algoritmalar calistirildi!")
print("=" * 60)
