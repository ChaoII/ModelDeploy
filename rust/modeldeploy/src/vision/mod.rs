pub mod detection;
pub mod classification;
pub mod face;
pub mod face_extra;
pub mod obb_iseg_pose;
pub mod ocr_lpr_attr;

pub use detection::UltralyticsDet;
pub use classification::UltralyticsCls;
pub use face::Scrfd;
pub use face_extra::{FaceRec, FaceAge, FaceGender, FaceAntiSpoofPipeline, FaceRecPipeline, FaceRecResult, Gender, AntiSpoofResult};
pub use obb_iseg_pose::{UltralyticsObb, UltralyticsSeg, UltralyticsPose, ObbResult, IsegResult, PoseResult};
pub use ocr_lpr_attr::{PaddleOcr, OcrRecognition, LprPipeline, PedestrianAttribute, OcrResult, LprResult, AttributeResult};
