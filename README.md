# The explanation of programs in this directory.  
<a name="contents"></a>
## Contents
### .sh
- [arangeDirectory.sh](#arangeDirectory.sh)
- [auto2DUnet3ch.sh](#auto2DUnet3ch.sh)
- [auto2DUnet6ch.sh](#auto2DUnet6ch.sh)
--- 

<a name="arangeDirectory.sh"></a>
## arangeDirectory.sh
This moves CT image, true label and predicted label to watch directory, which makes it easier to feed them into 3Dslicer.  
But, it is rarely used.  
<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="auto2DUnet3ch.sh"></a>
## auto2DUnet3ch.sh
This runs 2D U-Net program.  
When you run this, you are required to feed suffix which determine fed text file.  
suffix means ~ in text file, training_~.txt
It is used to determine validation file, name weight file best_suffix.hdf5 and so on.
<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>



<a name="subplot"></a>
## matplotのsubplotについての参考サイト
[サイト1](http://ailaby.com/matplotlib_fig/)

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>


<a name="shell"></a>
## シェルスクリプト参考サイト  
[サイト1](https://qiita.com/zayarwinttun/items/0dae4cb66d8f4bd2a337)

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="linux"></a>
## linux shortcut参考サイト
[サイト1](https://eng-entrance.com/linux_shortcut)

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="markdown"></a>
## markdown参考サイト
[サイト1](https://qiita.com/Qiita/items/c686397e4a0f4f11683d)  
[サイト2](https://qiita.com/kamorits/items/6f342da395ad57468ae3)

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="complement"></a>
## 画像補完を説明してくれているサイト
[サイト1](https://imagingsolution.blog.fc2.com/blog-entry-142.html)  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>


<a name="tf-fw"></a>
## tensorflowのFutureWarning
以下のようなwarningについて
```
 FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
```
**原因**  
numpy1.17.0がtensorflow1.14.0に対応していない
**対策**  
numpyのバージョンを1.16.4に落とせば良い  
(アンインストールしてからインストール)
```
sudo pip3 uninstall numpy
sudo pip3 install numpy==1.16.4
```
<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="tf-gpu"></a>
## tensorflow-gpuの導入

tensorflowでGPUを使えるようにするためには、大きく3つのことを行う必要がある(Ubuntu18.04)
1. Nvidia driverのインストール
1. tnsorflow-gpuのインストール
1. CUDAのインストール
1. CuDNNのインストール
1. tensorflowがGPUを認識しているか確認

### 1. Nvidia driverのインストール
**セキュアブートの無効化**  
一部のPCのグラボやハードウェア、OSを実行するためにセキュアブートを無効化する必要があるので、PCの電源を入れるときにF2やDelをおして、BIOSを呼び出す  
無効化については以下のサイトを参照  
[UEFI の セキュアブートの設定について: 変更方法や注意点など](https://freesoft.tvbok.com/tips/efi_installation/about_secure_boot.html)  
<br>
**nouveauの無効化**  
Linuxには、デフォルトでnouveauというドライバが使われているので、これを無効化する必要がある  
<br>
nouveauが使われているかどうかを確認
```
lsmod | grep nouveau
```
以下のような表示が出ればOK
```
nouveau              1403757  0
video                  24400  1 nouveau
mxm_wmi                13021  1 nouveau
i2c_algo_bit           13413  1 nouveau
drm_kms_helper        125008  1 nouveau
ttm                    93441  1 nouveau
drm                   349210  3 ttm,drm_kms_helper,nouveau
i2c_core               40582  5
drm,i2c_i801,drm_kms_helper,i2c_algo_bit,nouveau
wmi                    19070  2 mxm_wmi,nouveau
```

nouveauを無効化する
/etc/modprobe.d/blacklist-nvidia-nouveau.confを作成

```
vim  /etc/modprobe.d/blacklist-nvidia-nouveau.conf
```
以下の設定を記述
```
blacklist nouveau
options nouveau modeset=0
```
再読み込み
```
sudo update-initramfs -u
```
終了  
<br>
**Nvidia driverのインストール**
```
sudo ubuntu-drivers autoinstall
```
再起動
```
sudo reboot
```
nouveauが停止していることの確認  
```
lsmod | grep -i nouveau
```
何も出なければOK  
<br>
Nvidia driverが動作していることの確認  
```
nvidia-smi
```
以下のように表示されればOK  
```
Tue Sep  4 18:49:06 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.48                 Driver Version: 390.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 105...  Off  | 00000000:01:00.0  On |                  N/A |
| 52%   43C    P8    N/A /  75W |    166MiB /  4036MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1303      G   /usr/lib/xorg/Xorg                            96MiB |
|    0      1473      G   /usr/bin/gnome-shell                          67MiB |
+-----------------------------------------------------------------------------+
```

### 2. tensorflow-gpuのインストール
すでにtensorflowがインストールされている時は、それをアンインストールする  
```
sudo pip3 uninstall tensorflow
```
tensorflow-gpuをインストール(version:1.14.0)
```
sudo pip3 install tensorflow-gpu==1.14.0
```

### 3. CUDAのインストール
tensorflowのバージョンとマッチしたバージョンをもつCUDAのインストール  
以下のサイトで適切にクリックして、出てきたコマンドを実行  
(CUDA Toolkit 10.1 update2 Archive)[https://developer.nvidia.com/cuda-10.1-download-archive-update2]  
Linux -> x86_64 -> Ubuntu -> 18.04 -> deb(local)  
以下、上のようにクリックして出てきたコマンド  

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin  
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600  
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb  
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb  
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub  
sudo apt update  
sudo apt -y install cuda  
```
**パスを通す**  
(cudaなのかcuda-10.1なのかが不明)  
~/.bashrcを開く  
```
vim ~/.bashrc
```
~/.bashrcに以下の行を追加  
```
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
```
~/.bashrcを再読み込みする  
```
source ~/.bashrc
```

### 4. CuDNNのインストール
Nvidia developerに会員登録をする  
[Nvidia developerのサイト](https://developer.nvidia.com/cudnn)   
<br>
以下から、CUDA10.1に対応したCuDNNをダウンロードする  
[https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)  
<br>
ダウンロードするものは...  
Download cuDNN v7.6.4 (September 27, 2019), for CUDA 10.1の中にある  
cuDNN Runtime Library for Ubuntu18.04 (Deb)  
cuDNN Developer Library for Ubuntu18.04 (Deb)  
cuDNN Code Samples and User Guide for Ubuntu18.04 (Deb)  
の3つ 
<br>
ダウンロードしたディレクトリに移動し、以下のコマンドを用いてCuDNNをインストール
```
sudo dpkg -i libcudnn7_7.6.4.38-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.4.38-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-doc_7.6.4.38-1+cuda10.1_amd64.deb
```
終了

### 5. tensorflowがGPUを認識しているかどうかを確認
```
# Open python3
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```
以下のような、GPUが表示されればOK
```
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:00:04.0
totalMemory: 11.17GiB freeMemory: 11.10GiB
2018-01-12 20:09:07.452093: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 2319180638018740093
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 11324325888
locality {
  bus_id: 1
}
incarnation: 13854674477442207273
physical_device_desc: "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7"
]
```

認識されていないと以下のようにGPUが表示されない  
```
2019-12-03 01:02:22.615649: I tensorflow/core/platform/cpu_feature_guard.cc:145] This TensorFlow binary is optimized with Intel(R) MKL-DNN to use the following CPU instructions in performance critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 10517104128675153439
]
```

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="tf-error"></a>
## tensorflow-gpu導入時のエラー解決
tensorflow-gpuを導入し、tensorflowを動かしてみると、以下のようなエラーが出た。  
```
Could not dlopen library 'libcudart.so.10.0'; dlerror: libcudart.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:
```
cudaのライブラリが保存されている/usr/local/cuda/lib64の中身を見てみると、  
```
libaccinj64.so                libnppidei.so
libaccinj64.so.10.1           libnppidei.so.10
libaccinj64.so.10.1.243       libnppidei.so.10.2.0.243
libcudadevrt.a                libnppidei_static.a
libcudart.so                  libnppif.so
libcudart.so.10.1             libnppif.so.10
libcudart.so.10.1.243         libnppif.so.10.2.0.243
libcudart_static.a            libnppif_static.a
libcufft.so                   libnppig.so
libcufft.so.10                libnppig.so.10
libcufft.so.10.1.1.243        libnppig.so.10.2.0.243
libcufft_static.a             libnppig_static.a
libcufft_static_nocallback.a  libnppim.so
libcufftw.so                  libnppim.so.10
libcufftw.so.10               libnppim.so.10.2.0.243
libcufftw.so.10.1.1.243       libnppim_static.a
libcufftw_static.a            libnppist.so
libcuinj64.so                 libnppist.so.10
libcuinj64.so.10.1            libnppist.so.10.2.0.243
libcuinj64.so.10.1.243        libnppist_static.a
libculibos.a                  libnppisu.so
libcurand.so                  libnppisu.so.10
libcurand.so.10               libnppisu.so.10.2.0.243
libcurand.so.10.1.1.243       libnppisu_static.a
libcurand_static.a            libnppitc.so
libcusolverMg.so              libnppitc.so.10
libcusolverMg.so.10           libnppitc.so.10.2.0.243
libcusolverMg.so.10.2.0.243   libnppitc_static.a
libcusolver.so                libnpps.so
libcusolver.so.10             libnpps.so.10
libcusolver.so.10.2.0.243     libnpps.so.10.2.0.243
libcusolver_static.a          libnpps_static.a
libcusparse.so                libnvgraph.so
libcusparse.so.10             libnvgraph.so.10
libcusparse.so.10.3.0.243     libnvgraph.so.10.1.243
libcusparse_static.a          libnvgraph_static.a
liblapack_static.a            libnvjpeg.so
libmetis_static.a             libnvjpeg.so.10
libnppc.so                    libnvjpeg.so.10.3.0.243
libnppc.so.10                 libnvjpeg_static.a
libnppc.so.10.2.0.243         libnvrtc-builtins.so
libnppc_static.a              libnvrtc-builtins.so.10.1
libnppial.so                  libnvrtc-builtins.so.10.1.243
libnppial.so.10               libnvrtc.so
libnppial.so.10.2.0.243       libnvrtc.so.10.1
libnppial_static.a            libnvrtc.so.10.1.243
libnppicc.so                  libnvToolsExt.so
libnppicc.so.10               libnvToolsExt.so.1
libnppicc.so.10.2.0.243       libnvToolsExt.so.1.0.0
libnppicc_static.a            libOpenCL.so
libnppicom.so                 libOpenCL.so.1
libnppicom.so.10              libOpenCL.so.1.1
libnppicom.so.10.2.0.243      stubs
libnppicom_static.a

```
確かに、libcudart.so.10.0はなく、libcudart.so.10.1しかなかった。  
これは、CUDAのバージョンが合っていないことが原因。また、それに伴って、CuDNNのバージョンも下げる必要がある。  

### 前回のCUDAの削除
自分はやらなくてもできたが、新しいCUDAを入れる前にきれいにしておくといいかも。  
以下のコマンドを実行  
```
sudo apt purge cuda*
sudo apt purge nvidia-cuda-*
sudo apt purge libcuda*
sudo reboot
```

### CUDAのダウングレード
今回の場合は、libcudart.so.10.0を求められているので、CUDA10.0が必要。  
よって、[CUDAのサイト](https://developer.nvidia.com/cuda-10.0-download-archive)に移り、自分のマシンに合うように、選択し、表示されるコマンドを実行  
cudaのインストール時に、**バージョンを指定**する必要があった。  
```
× sudo apt install cuda
○ sudo apt install cuda=10.0
```
Linux -> x86_64 -> Ubuntu -> 18.04 -> deb(local)とした時、以下のようなコマンドを実行するよう言われた。  
```
sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```
最後の1行は、バージョンを指定する。  
これらを実行したのちに、以下のコマンドで、/usr/local確認し、    
```
ls /usr/local
```
cuda-10.0のディレクトリが存在していれば、cuda10.0をインストールできている  
その後、  
```
ls /usr/local/cuda
```
で、/usr/local/cudaの中身を確認し、libcudart.so.10.0が存在していれば、CUDAのダウングレードは成功しているはず。  
もし、存在していない場合は、PATHを/usr/local/cudaから/usr/local/cuda-10.0に変更(以下コマンド)
```
vim ~/.bashrc
```
以下を
```
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
```
次のように書き直す  
```
export PATH=/usr/local/cuda-10.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH}
```
```
source ~/.bashrc
```
で、bashrcを読み直せば、OK  
これに伴い、CuDNNもインストールし直す  
### CuDNNのインストール
Nvidia developerに会員登録をする  
[Nvidia developerのサイト](https://developer.nvidia.com/cudnn)   
<br>
以下から、CUDA10.0に対応したCuDNNをダウンロードする  
[https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)  
<br>
ダウンロードするものは...  
Download cuDNN v7.6.4 (September 27, 2019), for CUDA 10.0の中にある  
cuDNN Runtime Library for Ubuntu18.04 (Deb)  
cuDNN Developer Library for Ubuntu18.04 (Deb)  
cuDNN Code Samples and User Guide for Ubuntu18.04 (Deb)  
の3つ 
<br>
ダウンロードしたディレクトリに移動し、以下のコマンドを用いてCuDNNをインストール
```
sudo dpkg -i libcudnn7_7.6.4.38-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.4.38-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.6.4.38-1+cuda10.0_amd64.deb
```
この時、CuDNNは、上書きされるので、前回のものを消す必要はない。

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="nvidia-driver"></a>
## nvidia driverのインストール
ubuntuのサーバー版をインストールし、そこにnvidia driverを入れようとしたらubuntu-driversがないとエラーを吐かれたので、それのインストールを行った。(以下のコマンド)  
```
sudo apt isntall ubuntu-drivers-common
ubuntu-drivers devices # 差し込まれているグラボの情報が出るはず
```
そして、nvidia driverのインストール
```
sudo ubuntu-drivers autoinstall
reboot
```

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>


<a name="custom"></a>
## カスタム損失関数を使ったときのモデルロード
kerasでNNのモデルを構築したときに、損失関数や評価関数に自作のカスタム関数を使う時がある。  
そのモデルを保存し、再度ロードすると、以下のようなValueErrorが生じることがある。
```
ValueError: ('Unknown loss function', ':compute_loss')
```
**対策**  
モデルをロードする時(`tf.keras.models.load_model`)に、custom_objectsの引数にカスタム損失関数を渡す必要がある。評価関数も同様  
<br>
改善前  
```
import tensorflow as tf
from tf.keras.models import load_model

def custom_loss(ytrue, ypred):# 損失関数
    return ytrue - ypred

def custom_metrics(ytrue, ypred):# 評価関数
    return ytrue - ypred

model = load_model("model.hdf5")
```
改善後  
```
import tensorflow as tf
from tf.keras.models import load_model

def custom_loss(ytrue, ypred):
    return ytrue - ypred

def custom_metrics(ytrue, ypred):
    return ytrue - ypred

model = load_model("model.hdf5",
    custom_objects={'custom_loss':custom_loss, "custom_metrics"=custom_metrics)
```
<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="~"></a>
## ~/の処理(python)
pythonへあるファイルのパスを渡す時、~/dataなどで渡す時があった。  
しかし、~/はpythonでは認識されないので、以下の関数を用いて、~を絶対パスに変換する  
```python
import os
path = os.path.expanduser("~/path")
# /Usrs/tanimoto/path
```
また、シェルスクリプトを用いる時は、~を $HOMEに置き換えれば良い  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="git-lfs"></a>
## git lfsのインストール
以下のサイトの通りに進めると良い  
[git lfsのインストール](https://github.com/git-lfs/git-lfs/blob/master/INSTALLING.md#installing-packages)  
以下のコマンドでgit lfsのdebパッケージをダウンロード  
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
```
git lfsのインストール  
```
sudo apt install git-lfs
```
インストールできているかどうかの確認  
```
git lfs --version
```
以下のように出力されればOK  
```
git-lfs/2.7.2 (GitHub; darwin amd64; go 1.12.4)
```
また、使う前には以下のコマンドで、LFSを初期化した方が良い  
```
git lfs install
# Git LFS initialized.
```
<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="pbcopy"></a>
## pbcopyをubuntuで使いたい
pbcopyはMacのみのコマンドで、ターミナルの出力をクリップボードにコピーする  
例）
```
pbcopy
# 以下、標準入力
# aa
# bb
#Ctrl+Dで終了
# aa bbがクリップボードにコピーされる
```
以下のコードでは、test.txtの内容をクリップボードにコピー
```
cat test.txt | pbcopy
```
ubuntuでは、xselが同じような動きをする  
```
cat test.txt | xsel --clipboard --input
```
~/.bashrcに以下のaliasを書き込めば、pbcopyを使える  
```
alias pbcopy='xsel --clipboard --input'
```
<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="gpu-memory"></a>
## ubuntuでnvidiaのGPUメモリが解放されない
tensorflowなどで、学習を途中で止めたとき、以下のように何もプログラムを動かしていないのにGPUメモリがほとんど取られてしまっているケースがある。  
```
+------------------------------------------------------+
| NVIDIA-SMI    .       Driver Version:                |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 0000:05:00.0     Off |                  Off |
| N/A   32C    P8    26W / 149W |  12200MiB / 12287MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla K80           On   | 0000:06:00.0     Off |                  Off |
| N/A   29C    P8    29W / 149W |  12200MiB / 12287MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
プロセスが残ってしまっているのが原因らしいので、以下のコマンドでそれらを確認  
```
lsof /dev/nvidia*
```
出力結果
```
COMMAND    PID         USER   FD   TYPE DEVICE  SIZE/OFF NODE NAME
cdpd       354 tanimotoryou    0r   CHR    3,2       0t0  313 /dev/null
cdpd       354 tanimotoryou    1u   CHR    3,2       0t0  313 /dev/null
cdpd       354 tanimotoryou    2u   CHR    3,2       0t0  313 /dev/null
commerce   355 tanimotoryou    0r   CHR    3,2       0t0  313 /dev/null
commerce   355 tanimotoryou    1u   CHR    3,2       0t0  313 /dev/null
commerce   355 tanimotoryou    2u   CHR    3,2       0t0  313 /dev/null
Categorie  357 tanimotoryou    0r   CHR    3,2       0t0  313 /dev/null
Categorie  357 tanimotoryou    1u   CHR    3,2       0t0  313 /dev/null
Categorie  357 tanimotoryou    2u   CHR    3,2       0t0  313 /dev/null
Dock       361 tanimotoryou    0r   CHR    3,2       0t0  313 /dev/null
Dock       361 tanimotoryou    1u   CHR    3,2       0t0  313 /dev/null
Dock       361 tanimotoryou    2u   CHR    3,2       0t0  313 /dev/null
```
このうち、使っていないプロセスをkillコマンドで消す
```
kill -9 (該当するPID)
```

## killコマンドについて
使い方  
```
kill [option] (プロセスID)
```
**主なオプション**
- -s シグナル or - シグナル : 指定したシグナル名かシグナル番号を送信  
- -l : シグナル名とシグナル番号の対応の一覧を表示  

**主なシグナル**  
|シグナル番号|シグナル名|動作| 
|---|---|---|
|1  |SIGHUP   |再起動        |
|6  |SIGABRT  |中断          |
|9  |SIGKILL  |強制終了      |
|15 |SIGTERM  |終了(default) |
|17 |SIGSTOP  |停止          |
|18 |SIGCONT  |再開          |

また、シグナル名を指定する時は、最初の3文字を省略して入力  
例）SIGABRTシグナルを送る時  
```
kill -ABRT プロセスID
```

シグナルの一覧表示  
```
kill -l
 1) SIGHUP	 2) SIGINT	 3) SIGQUIT	 4) SIGILL
 5) SIGTRAP	 6) SIGABRT	 7) SIGEMT	 8) SIGFPE
 9) SIGKILL	10) SIGBUS	11) SIGSEGV	12) SIGSYS
13) SIGPIPE	14) SIGALRM	15) SIGTERM	16) SIGURG
17) SIGSTOP	18) SIGTSTP	19) SIGCONT	20) SIGCHLD
21) SIGTTIN	22) SIGTTOU	23) SIGIO	24) SIGXCPU
25) SIGXFSZ	26) SIGVTALRM	27) SIGPROF	28) SIGWINCH
29) SIGINFO	30) SIGUSR1	31) SIGUSR2
```
<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="ssh"></a>
## 秘密鍵、公開鍵の作成
これらの鍵は、~/.sshというディレクトリに保存するのが一般的なので、そのディレクトリがなければ、作成し、そこに移動  
```
# ~/.sshの作成
mkdir ~/.ssh
# ~/.sshに移動
cd ~/.ssh
```
以下のコマンドを実行すると、秘密鍵、公開鍵が作成される  
```
ssh-keygen -t rsa
```
-tは、鍵の暗号化形式を指定する(dsa, edcsa, ed25519から選ぶ)  
上のコマンドを実行すると、以下のように出力される  
```
Generating public/private rsa key pair.
Enter file in which to save the key (/Users/(username)/.ssh/id_rsa):
```
ここで、鍵の名前を指定する。(defaultはid_rsa)
その後、以下のように出力されるので、Enterを2回押すと、~/.ssh内に鍵が生成される  
```
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
```
~/.sshのなかを確認する
```
ls ~/ssh
```
以下のようなものがあればOK  
```
id_rsa id_rsa.pub
# id_rsaの部分は、自分で決めた名前
```
id_rsaは、秘密鍵なので、誰にも知られてはいけない  
id_rsa.pubを接続先に渡してssh接続を行う  


<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="ssh-connection"></a>
## LAN内ssh接続
まず、ssh接続するために必要なもの
1. 接続先のアカウントとパスワード
1. 接続先のローカルIPアドレス
のちに必要になるもの  
1. 秘密鍵、公開鍵

### 1. 接続先のローカルIPアドレスを確認する
接続先で、以下のコマンドを打つ  
```
ip addr
か
ifconfig
```
すると、以下のような出力が出る(ifconfig)  
```
eth0      Link encap:Ethernet(1)  HWaddr 00:80:90:44:08:11(2)
          inet addr:192.168.1.11(3)  Bcast:192.168.1.255(4)  Mask:255.255.255.0(5)
          inet6 addr: fe80::3199:ff:fe01:3762/64 Scope:Link(6)
          UP(7) BROADCAST(8) RUNNING(9) MULTICAST(10)  MTU:1500(11)  Metric:1(12)
          RX packets:583312 errors:0 dropped:0 overruns:0 frame:0(13)
          TX packets:28344 errors:0 dropped:0 overruns:0 carrier:0(13)
          collisions:0 txqueuelen:100(13)
          RX bytes:4987886272 (4.9 GB)  TX bytes:50440257 (50.4 MB)(14)
eth1      Link encap:Ethernet  HWaddr 00:80:00:48:AA:88
          BROADCAST MULTICAST  MTU:1500  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:100
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)
lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
          UP LOOPBACK RUNNING  MTU:3924  Metric:1
          RX packets:16 errors:0 dropped:0 overruns:0 frame:0
          TX packets:16 errors:0 dropped:0 overruns:0 carrier:0
```
eth0などをインターフェイス名と呼ぶ。設定時に指定するための名称。  
また、loはローカルループバックと呼ばれる特別な仮想インターフェイスで、仮想的なネットワークテストで使えるように用意されているので、これを用いて外部と通信はできない。  
lo以外で、inetが割り当てられているのが、正常に起動しているもの（なはず）  
以下、eth0やloの参考サイト  
[https://www.atmarkit.co.jp/ait/articles/0109/29/news004.html](https://www.atmarkit.co.jp/ait/articles/0109/29/news004.html)  
今回の場合は、192.168.1.11、サブネットマスクが255.255.255.0  

### 2. ssh接続を行う
sshを行うためには、以下のコマンド  
```
ssh acount@192.168.1.11
とか
ssh -l acount 192.168.1.11
とか
ssh 192.168.1.11
```
そうすると、以下のような出力がされ、yesかnoを答えるように言われる  
```
The authenticity of host '10.0.2.4 (10.0.2.4)' can't be established.
ECDSA key fingerprint is SHA256:jjXyMlozEF8NMOjMTfstVsS1QWwtnd6bmiKgm+dRWJY.
Are you sure you want to continue connecting (yes/no)?
```
この時、求める接続先に接続できているかを確認するために、接続先側で、以下のコマンドを打ち込む  
```
ssh-keygen -lf /etc/ssh/ssh_host_ecdsa_key.pub
```
この時出力されるfingerprint（長い文字列)がssh接続した時に出てきたfingerprintと一致するかを確認し、一致すれば、yesと入力する。  
そして、アカウントとパスワードを入力すると、以下のように出力され、ssh接続できるようになる。  
```
Last login: Fri Jun 15 06:48:46 2018
yazaki@ubuntu-server:~$
```
exitと入力すると、切断できる。  
もし接続できない時は、接続先でファイアウォールが有効になって接続を弾かれている可能性があるので、ssh接続を行うために、22番ポートを解放する必要がある。(以下参考)  
[ファイアウォールの設定(ufw)](#ufw)  

### 3. セキュリティを高める
デフォルトだと、セキュリティがガバガバなので、以下の2つを行い、セキュリティを高める。  
1. 接続先がsshを受け入れるポート番号の変更
1. 秘密鍵、公開鍵を用いて認証する

#### 1. 接続先のsshを受け入れるポート番号の変更
デフォルトでは、sshは22番ポートを利用するが、有名なので少し危険。そこで、使われていないポート番号をsshのポートに変更することで、セキュリティを高める。自分の場合は、22222を使った。(1~65535番からえらぶ)  

**ポート番号の変更**  
接続先で以下のコマンドをうち、sshd_configを開いて、設定を書き込む。  
```
sudo vim /etc/ssh/sshd_config
```
ここから、Port書かれた行を探し、#がついてたら外して、22と書かれている部分を先ほど決めた新しいポート番号に書き換える。  
以下のコマンドで、sshdをリスタート。  
```
sudo service sshd restart
```
これでポート番号が変更されたはず。  

**ポート番号が変更されたことの確認**
先ほどと同じように、ssh接続を試みると、refusedされる。
```
ssh 192.168.1.11
```
出力  
```
ssh: connect to host 192.168.1.11 port 22: Connection refused
```
ポート番号を指定すると、ログインできる。(-pの後ろに新しいポート番号)
```
ssh 192.168.1.11 -p 22222
```
そうすると、ログインできる。
ここで、ログインできない時は、ファイアウォールが原因であることがあるので、新しいポート番号を接続先側で解放する手順をとる必要がある。そのためにポート番号を22に書き直し、ログインする必要がある。  
[ポート解放](#ufw)  

#### 2. 秘密鍵、公開鍵を用いて認証
接続元で、鍵を作成する。(接続先では作成しないこと)鍵作成は、以下を参考に  
[秘密鍵、公開鍵の作成](#ssh)  

作成した公開鍵を接続先に送信するために以下のコマンドを打つ  
```
scp -P 22222 ~/.ssh/id_ecdsa.pub yazaki@10.0.2.4:~/id_ecdsa_yazaki-mint.pub
```
~/.ssh/id_ecdsa.pubは、接続元で作った公開鍵のパス。  
~/id_ecdsa_yazaki-mint.pubは、公開鍵を送信した時の保存先のパス。(鍵の名前は自由に設定可能)  

これを行ったら、sshで接続先に接続。
```
ssh 192.168.1.11 -p 22222
```
以下のコマンドを打ち込み、公開鍵を有効にする。
```
mkdir .ssh
chmod 700 .ssh
cat id_ecdsa_yazaki-mint.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_leys
```
複数のPCから同じ接続先に接続したい時は、公開鍵を~/.ssh/authorized_keysに追記(>>)していけば良い。（なお、その時は3行目のみでOK）  

これを行った後に、接続し直してみる  
```
ssh 192.168.1.11 -p 22222
```
パスフレーズを求められた時は、入力する。  
ただ、このままでは結局パスワードがわかっていればログインできてしまい、鍵の意味がないので、パスワードによるログインを不可にする。  
以下のコマンドでsshd_configを接続先で開く。  
```
sudo vim /etc/ssh/sshd_config
```
PasswordAuthenticationの行を探し、PasswordAuthentication noに書き換える。(#がついてたら、外す)  
sshdをリスタートし、ログインすると、パスワードを求められることなくログインできるはず。（秘密鍵を持っていなかったら、denidされる)
```
sudo service sshd restart
```
```
ssh 192.168.1.11 -p 22222 
```

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>


<a name="dual"></a>
## Ubuntu18.04を別ディスクでデュアルブート
この方法では、SSDなどを新しく用意しそこに新たなOSを入れることで、パーティション分割不要で簡単にできる。  
大きく、行うことは3つ  
1. Ubuntu18.04のブートができるUSBを用意する
1. デスクトップPCの準備
1. Ubuntuのインストール  

### 1. Ubuntu18.04のブートができるUSBを用意する
適当なUSBを用意する。
windows10、Ubuntu18.04それぞれでの方法を説明。 
#### windows10から
**ubuntuのダウンロード**  
以下のサイトから、ubuntu-18.04.1-desktop-amd64.isoをダウンロード。  
[Ubuntu18.04のダウンロードサイト](http://jp.releases.ubuntu.com/18.04.3/)

**USBメモリのフォーマット**  
新しいUSBメモリであれば必要ないかもしれないが、念のため、やっておくといいかも。  
以下のサイトを参考。  
[フォーマットのサイト](https://www.iodata.jp/support/qanda/answer/s16612.htm)  
**USBメモリへの書き込み**  
ただ、isoファイルをコピーするだけではダメで、特別なことをしてisoファイルをLiveCDとしてUSBメモリに書き込む必要がある。 
Unetbootinというアプリをインストール、使用して書き込む。  
[Unetbootin](https://unetbootin.github.io)  
Unetbootinを開き、ディスクイメージに、ダウンロードしたisoファイルを指定し、タイプをUSBドライブにしてOKをおす。また、ドライブが、さしたUSBになっていることを確認。

#### ubuntuから
**ubuntuのインストール**  
以下のコマンドで、ubuntuのisoファイルをダウンロード。  
```
curl http://jp.releases.ubuntu.com/18.04.3/ubuntu-18.04.3-desktop-amd64.iso 
```
また、ubuntuで大きなファイルをダウンロードする時は、wgetやcurlよりもaxelがいいらしい。(以下参考)
[axelの参考](#axel)  

**USBメモリへの書き込み**  
USBを差し込む前後で以下のコマンドを実行し、差し込んだUSBのディスク名を確認。  
```
sudo fdisk -l
```
**USBのフォーマット**  
念のため、フォーマット  
以下のコマンドを打つだけ  
```
mkfs -t fat32 /deb/sdb1
```
-tのあとは、フォーマットの形式。fat32にする必要がある。  
/dev/sdb1は、先ほど確認したディスク名。  

**USBメモリへの書き込み**  
ddというコマンドで、書き込める。（時間がかかるかも）  
その前に、USBをアンマウントしなければいけないので、以下のコマンドを実行  
その前に、アンマウント先を再確認
```
lsblk -f
```
出力例  
```
NAME   FSTYPE LABEL     UUID                                 MOUNTPOINT
...
sdf                                                          
└─sdf1 vfat   TRANSCEND 1B17-B118                            /run/media/zero/TRANSCEND
...
```
USBをアンマウント  
```
umount /run/media/zero/TRANSCEND
```
umount以下は、lsblkで確認した MOUNTPOINTに書かれているパス。  
ddで書き込み。  
dd bs=4M if=/home/zero/Downloads/ubuntu-ja-16.04-desktop-amd64.iso of=/dev/sdf status=progress && sync
```
if以下は、先ほどダウンロードしたisoファイルのパス。  
of以下は、先ほど確認したUSBのディスク名（数字なし）  
```
LiveUSBの完成。  

### 2. デスクトップPCの準備
windowsのブートドライブがつながっていても、[うまくやる方法](https://www.g104robo.com/entry/ubuntu-dualboot-win10-uefi)はあるが、操作ミスをすると、windowsのドライブにubuntuを書き込んでしまう可能性があるので、Ubuntuを入れるSSD以外は物理的に外しておくと安心。  
PCを再起動して、BIOSを立ち上げ（立ち上げるときにDELとかF2とか）、セキュアブートの無効化と、ファーストブートの無効化、Boot順序の設定を行う。  
設定画面は、PCのメーカーによって異なるので、以下のサイトのUEFI BIOSでの設定というところを見ながら、対応したところをいじっていくとよい。  
[UEFI](https://www.g104robo.com/entry/ubuntu-dualboot-win10-uefi)

### 3.ubuntuのインストール
2.を行った後、再起動されると、ubuntuがインストールされるので、適当に選択していく。  

インストールし終わったら、物理的に抜いていたwindowsのドライブやその他ドライブを付け直す。  

終了  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="axel"></a>
## Ubuntuで大きなファイルをダウンロードする時(axel)
Linuxでのファイルダウンロードには、wgetやcurlが使われることが多いが、大きなファイルをダウンロードする時は、多重接続により、5倍以上のスピードでダウンロードできることがある。それが、axelだ。  
**axelのインストール**  
```
sudo apt install axel
```
**axelの使い方**  
```
axel -a http://jp.releases.ubuntu.com/18.04.3/ubuntu-18.04.3-desktop-amd64.iso
```
-aは、分割状況をわかりやすくしてくれるオプション。  
ダウンロードを中断すると、ダウンロード名に.stがついたファイルが残り、これにダウンロード状況が保存されているので、再実行すると、続きからダウンロードされる。  
以下がaxelのヘルプ。
```
axel --help
Usage: axel [options] url1 [url2] [url...]

--max-speed=x           -s x    Specify maximum speed (bytes per second)
--num-connections=x     -n x    Specify maximum number of connections
--output=f              -o f    Specify local output file
--search[=x]            -S [x]  Search for mirrors and download from x servers
--header=x              -H x    Add header string
--user-agent=x          -U x    Set user agent
--no-proxy              -N      Just don't use any proxy server
--quiet                 -q      Leave stdout alone
--verbose               -v      More status information
--alternate             -a      Alternate progress indicator
--help                  -h      This information
--version               -V      Version information

Visit https://github.com/eribertomota/axel/issues to report bugs
```

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="DNS"></a>
## Could not resolve host:github.com
sshでgithubにログインしようとした時、題名のようなエラーを吐かれた。  
グローバルIPで試すと、入れるときはDNSのキャッシュをクリアする必要がある。(mac)

基本的には、再起動すれば良いらしいが、再起動できない時などは以下のコマンドを打てばいいらしい。  
```
sudo killall -HUP mDNSResponder
```
以下を参考に  
[https://support.apple.com/ja-jp/HT202516](https://support.apple.com/ja-jp/HT202516)

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="ufw"></a>
## linuxにおけるファイアウォールの設定
linuxでファイアウォールの設定を行いたい時には、ufwを使うと良い。  
**ufwのインストール**  
```
sudo apt install ufw
```
ファイアウォールが有効になっているかの確認  
```
sudo ufw status
```
以下のように出力される  
```
Status: active #active:有効

To                         アクションFrom
--                         -------------------
OpenSSH                    LIMIT   Anywhere
```
**ファイアウォールを有効にする**
```
sudo ufw enable
```
**ファイアウォールを無効にする**
```
sudo ufw disable
```
**ポート番号Nを解放**
```
sudo ufw allow N
sudo ufw allow 22
```

**ポート番号Nを閉める**
```
sudo ufw delete N
sudo ufw delete 22
```

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="mount"></a>
## linuxにおけるUSBのマウント方法
linuxサーバなどでUSBを利用したい時は、windowsと違い、手動でマウント、アンマウントしなければいけない。  
<br>
**マウント**  
以下のコマンドで、デバイスが認識されていることを確認する。  
```
sudo dmesg
```
多くの場合は、sdb1らしい。  
マウントしてみる。  
マウントは、「/」以下のいずれかのディレクトリの情報をusbに割り当てることをする。  
逆に言えば、マウントで指定されたディレクトリに含まれている情報は、usbに保存される。マウントは、USBとディレクトリを繋ぐために行う。  
一般的には、/mntや/mediaにマウントすることが多いので、それらのディレクトリがない時は、mkdirで作る。  
以下がマウントのコマンド  
```
sudo mount /dev/sdb1 /mnt
```
```
ls /mnt
```
をしてみると、   
```
System Volume Information
```
記憶媒体の中身が見えるようになる。（保存されているものとか）  
次に以下のコマンドで、どのデバイスがどこにマウントされているかを確認する。  
```
sudo df -Th
```
```
/dev/sdb1 vfat 7.3G 128K 7.3G 1% /mnt
```
が出力されると、/dev/sdb1が/mntにマウントされていることが確認できたことになる。  

```
sudo touch /mnt/test.txt
```
とすると、正常に書き込まれ、
```
ls /mnt
```
とすると、
```
System Volume Information test.txt
```
test.txtが増えている。つまり、ちゃんと書きこめていることがわかる。  
<br>
**アンマウント**  
USBを取り外す前に必ず行う。  
```
umount /mnt(もしくは、パーティション名/dev/sdb1)
```
```
df -Th
```
で、
```
/dev/sdb1 vfat 7.3G 128K 7.3G 1% /mnt
```
が無ければ、アンマウントされている。
しかし、umountは、OS上で記憶媒体として扱わなくなっただけで、ハードウェア的にはつながったままなので、USBを取り出すには以下のejectを行う必要がある。
```
sudo eject /dev/sdb
```
/dev/sdbには、デバイス名を入力  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="sdb"></a>
## sdbとsdb1の違い
sdbは、記憶媒体の名前を表し、sdb1, sdb2,...は、その記憶媒体の中のパーティションを表す。  
**パーティション**  
パーティションとは、1つの記憶媒体を複数の記憶媒体で構成されているかのようにするために分けられてた記憶領域のこと。パーティションを分け、それぞれにデータとOSを別々に保存することで、片方に障害が起きても、もう一方に被害が広がらないようになる。また、1つの記憶媒体に複数のOSをインストールすることも可能。 
よって、複数に分割されているUSBをマウントしたい時は、パーティションごとに、マウントをする必要がある。  
[参考サイト](https://allabout.co.jp/gm/gc/438839/)  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="freeze"></a>
## ubuntuがフリーズ
デスクトップが固まってしまった時は、Ctrl + Alt + F2(F3, F4)でCUIを立ち上げる。topコマンドで過負荷なプロセスを特定し、PIDを確認。  
```
kill -9 PID
```
でプロセスを強制終了。  
Ctrl + Alt + F7(F2, F1)でGUIに戻る。  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="usb"></a>
## USBの初期化
USBを初期化することで、LiveUSBを普通に使い直すことができる。  
```
lsblk
```
と入力することで、PCに接続されているデバイスが一覧表示される。  
容量などから、初期化したいUSBを探す。  
見つけたら、アンマウントする。  
```
sudo umount /dev/sdb1(もしくは、マウントポイント)
```
**USBメモリ内の全てのデータを消去する。**
```
sudo dd if=/dev/zero of=/dev/sdb bs=4k status=progress && sync
```
if以下のdev/zeroはNULLがたくさん入っているファイル。  
of以下のdev/sdbは、デバイス名を書き込む。  
<br>
**USBの新しいパーティションテーブルを作成**
```
sudo fdisk /dev/sdb
```
と入力して、Enter、Oを押して、からのパーティションテーブルを作成する。  
Nを押して、新規パーティションの作成。  
プライマリパーティションを作成するために、pをおす。  
1を押して、パーティション1を作成。（これがプライマリパーティションになる）  
ファーストセクタとラストセクタはデフォルト値でいいので、Enterをおす。  
wで書き込み結果を保存。(時間がかかるかもしれない)  
<br>
**プライマリパーティション**  
OSの導入や起動ができるもの。(LiveUSBには必須)一般に1台で4つまで設けられる。  
```
lsblk
```
上のコマンドで、新しく作成したパーティションを確認する。  
```
sdb
 sdb1
```
のようにパーティション名が現れればOK  
<br>
**フォーマットする**  
```
sudo mkfd.vfat /dev/sdb1
```
vfatは、FAT32を意味する。  
/dev/sdb1は、パーティションラベルを入力。  
<br>
**USBを取り出す**
```
sudo eject /dev/sdb
```
<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>


