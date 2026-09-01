param([string]$Key = "")
$root = "E:\CLionProjects\ModelDeploy"
$data = "$root\test_data\test_models"
$img  = "$root\test_data\test_images"
$conf = "$root\tools\convert"
$tc   = "$root\tools\docker\sophgo"
$dockerArgs = @("run", "--rm", "-v", "${data}:/conv", "-v", "${img}:/cali_img",
                "-v", "${conf}:/tconf", "-v", "${tc}:/tc",
                "tpuc_dev:1.27-slim", "bash", "/tconf/conv_in_docker.sh")
if ($Key) { $dockerArgs += $Key }
& docker @dockerArgs 2>&1
